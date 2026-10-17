"""Real-time dual-fisheye stitch.

Each frame is projected with the shared fisheye model. A small seam state is
then updated from the overlap only:

1. Horizontal and vertical shift, from normalized correlation on the overlap.
2. A global exposure gain plus a smoothed per-row gain, so the seam brightness matches.
3. A short feather, so the lighting change is spread across the overlap.

Re-solving lens distortion every frame is too slow for live video and is not
what makes the seam jump. The shift absorbs a small residual yaw or pitch.
The gains absorb exposure and edge falloff.

``tests/test_realtime_stitch.py`` scores the fraction of seam pixels that land
within 8 gray levels of the source panorama. The synthetic cases require at
least 99 percent. That score is for this controlled overlap, not a promise for
every real photograph.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StitchEstimate:
    """Seam state applied to the latest frame."""

    shift_x: float
    shift_y: float
    gain: float
    overlap_px: int


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


class RealtimeStitcher:
    """Project a side-by-side dual fisheye and correct the seam every frame.

    Parameters
    ----------
    lens_fov_deg:
        Fisheye field of view across the disc diameter, in degrees.
        Values above 180 leave an overlap the solver can lock onto.
    base_fov_deg:
        Longitude kept from the middle of each lens in the output.
    output_width:
        Equirectangular width. Defaults to the input frame width.
    smoothing:
        Exponential weight on the previous seam estimate, in ``[0, 1)``.
        ``0`` applies the new measurement immediately.
    """

    def __init__(
        self,
        lens_fov_deg: float = 200.0,
        base_fov_deg: float = 180.0,
        output_width: int | None = None,
        smoothing: float = 0.4,
    ) -> None:
        if lens_fov_deg <= base_fov_deg:
            raise ValueError("lens_fov_deg must exceed base_fov_deg so the discs overlap")
        self.lens_fov_deg = float(lens_fov_deg)
        self.base_fov_deg = float(base_fov_deg)
        self.output_width = output_width
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.gain = 1.0
        self._maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.gain = 1.0

    def stitch(self, frame: np.ndarray) -> tuple[np.ndarray, StitchEstimate]:
        """Stitch one side-by-side dual-fisheye frame to equirectangular BGR."""
        frame = _as_bgr(frame)
        height, width = frame.shape[:2]
        left = frame[:, : width // 2]
        right = frame[:, width // 2 :]
        return self.stitch_pair(left, right)

    def stitch_pair(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, StitchEstimate]:
        left = _as_bgr(left)
        right = _as_bgr(right)
        out_w = self.output_width or (left.shape[1] * 2)
        out_h = out_w // 2
        half_w = out_w // 2
        proj_w = max(half_w + 2, int(round(half_w * self.lens_fov_deg / self.base_fov_deg)))
        overlap = proj_w - half_w

        left_patch = self._project(left, proj_w, out_h)
        # Match the existing dual-fisheye convention: mirror the right disc
        # before and after the shared forward projection.
        right_patch = np.fliplr(self._project(np.fliplr(right), proj_w, out_h))

        dx, dy = self._measure_shift(left_patch, right_patch, overlap)
        dx, dy = self._smooth(dx, dy)
        shifted = _warp(right_patch, dx, dy)

        gain, row_gain = self._measure_gain(left_patch, shifted, overlap)
        self.gain = (1.0 - self.smoothing) * gain + self.smoothing * self.gain
        corrected = _apply_gain(shifted, self.gain, row_gain)

        stitched = _composite(left_patch, corrected, half_w, overlap)
        estimate = StitchEstimate(self.shift_x, self.shift_y, self.gain, overlap)
        return stitched, estimate

    def _smooth(self, dx: float, dy: float) -> tuple[float, float]:
        a = self.smoothing
        self.shift_x = (1.0 - a) * dx + a * self.shift_x
        self.shift_y = (1.0 - a) * dy + a * self.shift_y
        return self.shift_x, self.shift_y

    def _project(self, fisheye: np.ndarray, proj_w: int, proj_h: int) -> np.ndarray:
        map_x, map_y = self._fisheye_maps(fisheye.shape[1], fisheye.shape[0], proj_w, proj_h)
        return cv2.remap(
            fisheye,
            map_x,
            map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _fisheye_maps(
        self, src_w: int, src_h: int, proj_w: int, proj_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (src_w, src_h, proj_w, proj_h, round(self.lens_fov_deg, 4))
        cached = self._maps.get(key)
        if cached is not None:
            return cached

        radius = min(src_w, src_h) / 2.0
        cx = src_w / 2.0
        cy = src_h / 2.0
        vs, us = np.mgrid[0:proj_h, 0:proj_w]
        lon_range = math.radians(self.lens_fov_deg)
        lon = (us / proj_w) * lon_range - lon_range / 2.0
        lat = (0.5 - vs / proj_h) * math.pi
        xw = np.cos(lat) * np.cos(lon)
        yw = np.cos(lat) * np.sin(lon)
        zw = np.sin(lat)
        px, py, pz = -zw, yw, xw
        phi = np.arccos(np.clip(pz, -1.0, 1.0))
        theta = np.arctan2(py, px) - math.pi / 2.0
        radial = phi / math.radians(self.lens_fov_deg / 2.0)
        map_x = (cx + radial * np.cos(theta) * radius).astype(np.float32)
        map_y = (cy - radial * np.sin(theta) * radius).astype(np.float32)
        self._maps[key] = (map_x, map_y)
        return map_x, map_y

    def _measure_shift(
        self, left_patch: np.ndarray, right_patch: np.ndarray, overlap: int
    ) -> tuple[float, float]:
        """Shift that moves the right patch onto the left patch in the overlap.

        A bounded normalized-correlation search is used instead of a full-frame
        phase correlation. The overlap is only a few dozen pixels wide, and a
        global FFT peak there locks onto the latitude wrap instead of the seam.
        """
        if overlap < 16:
            return 0.0, 0.0
        left_gray = _gray(left_patch[:, -overlap:])
        right_gray = _gray(right_patch[:, :overlap])
        row_ok = (left_gray.mean(axis=1) > 2.0) & (right_gray.mean(axis=1) > 2.0)
        if int(row_ok.sum()) < 24:
            return 0.0, 0.0
        idx = np.flatnonzero(row_ok)
        y0, y1 = int(idx[0]), int(idx[-1]) + 1
        left_gray = left_gray[y0:y1]
        right_gray = right_gray[y0:y1]
        max_shift = min(12, overlap // 4)
        if max_shift < 1 or left_gray.shape[0] <= 2 * max_shift + 8:
            return 0.0, 0.0
        template = right_gray[max_shift:-max_shift, max_shift:-max_shift]
        scores = cv2.matchTemplate(left_gray, template, cv2.TM_CCOEFF_NORMED)
        _score, _max_val, _min_loc, loc = cv2.minMaxLoc(scores)
        # With no shift the template origin is (max_shift, max_shift).
        dx = float(loc[0] - max_shift)
        dy = float(loc[1] - max_shift)
        return dx, dy

    def _measure_gain(
        self, left_patch: np.ndarray, right_patch: np.ndarray, overlap: int
    ) -> tuple[float, np.ndarray]:
        band = max(8, min(overlap, 48))
        left_gray = _gray(left_patch[:, -band:])
        right_gray = _gray(right_patch[:, :band])
        valid = (left_gray > 2.0) & (right_gray > 2.0)
        if int(valid.sum()) < 32:
            return 1.0, np.ones(left_patch.shape[0], np.float32)
        gain = float(np.median(left_gray[valid]) / max(float(np.median(right_gray[valid])), 1e-3))
        gain = float(np.clip(gain, 0.3, 3.0))

        row = np.ones(left_gray.shape[0], np.float32)
        for y in range(left_gray.shape[0]):
            m = valid[y]
            if int(m.sum()) < 6:
                continue
            row[y] = np.median(left_gray[y, m]) / max(float(np.median(right_gray[y, m])), 1e-3)
        # Remove the global gain so the row profile is only the edge falloff.
        row = row / max(gain, 1e-3)
        row = np.clip(row, 0.5, 2.0)
        row = cv2.GaussianBlur(row.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=6).ravel()
        row = row.astype(np.float32)
        # Ignore resampling noise. A real exposure or edge-falloff error moves
        # the seam by much more than this.
        if abs(gain - 1.0) < 0.04 and float(np.max(np.abs(row - 1.0))) < 0.06:
            return 1.0, np.ones(left_patch.shape[0], np.float32)
        return gain, row


def _warp(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return image
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_gain(image: np.ndarray, gain: float, row_gain: np.ndarray) -> np.ndarray:
    scale = (gain * row_gain).astype(np.float32).reshape(-1, 1, 1)
    corrected = np.clip(image.astype(np.float32) * scale, 0.0, 255.0)
    return corrected.astype(np.uint8)


def _composite(
    left_patch: np.ndarray, right_patch: np.ndarray, half_w: int, overlap: int
) -> np.ndarray:
    """Place each lens in its hemisphere and feather both vertical seams.

    Left patch column ``c`` lands on output ``c - margin``. Right patch column
    ``c`` lands on output ``half_w - margin + c``, so the two overlap bands
    meet at the center seam and again at the wrap-around edges.
    """
    margin = overlap // 2
    out_h = left_patch.shape[0]
    out = np.zeros((out_h, half_w * 2, 3), np.float32)
    weight = np.zeros((out_h, half_w * 2), np.float32)
    _accumulate(out, weight, left_patch, x0=-margin)
    _accumulate(out, weight, right_patch, x0=half_w - margin)
    valid = weight > 0
    out[valid] /= weight[valid, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _accumulate(out: np.ndarray, weight: np.ndarray, patch: np.ndarray, x0: int) -> None:
    """Feather the outer `overlap` columns of a patch into the canvas."""
    out_h, out_w = weight.shape
    proj_w = patch.shape[1]
    # Feather width follows the extra columns past a hemisphere.
    feather = max(1, proj_w - (out_w // 2))
    ramp = np.ones(proj_w, np.float32)
    ramp[:feather] = np.linspace(0.0, 1.0, feather, endpoint=False)
    ramp[-feather:] = np.linspace(1.0, 0.0, feather, endpoint=False)
    patch_f = patch.astype(np.float32)
    for c in range(proj_w):
        x = x0 + c
        if x < 0 or x >= out_w:
            continue
        column = patch_f[:, c]
        mask = column.sum(axis=1) > 0
        w = float(ramp[c])
        out[mask, x] += column[mask] * w
        weight[mask, x] += w


def render_fisheye_from_equirect(
    equirect: np.ndarray, fisheye_size: int, fov_deg: float
) -> np.ndarray:
    """Inverse of the stitcher's projection, used to build synthetic discs.

    ``equirect`` is the projection canvas: longitude runs from ``-fov/2`` to
    ``+fov/2`` across its width, latitude from ``+90`` to ``-90`` down its height.
    """
    proj_h, proj_w = equirect.shape[:2]
    radius = fisheye_size / 2.0
    cx = cy = radius
    ys, xs = np.mgrid[0:fisheye_size, 0:fisheye_size]
    dx = xs - cx
    dy = cy - ys
    radial = np.sqrt(dx * dx + dy * dy) / radius
    theta = np.arctan2(dy, dx)
    phi = radial * math.radians(fov_deg / 2.0)
    alpha = theta + math.pi / 2.0
    px = np.sin(phi) * np.cos(alpha)
    py = np.sin(phi) * np.sin(alpha)
    pz = np.cos(phi)
    xw, yw, zw = pz, py, -px
    lon = np.arctan2(yw, xw)
    lat = np.arcsin(np.clip(zw, -1.0, 1.0))
    lon_range = math.radians(fov_deg)
    map_x = ((lon + lon_range / 2.0) / lon_range * proj_w).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * proj_h).astype(np.float32)
    disc = radial <= 1.0
    image = cv2.remap(
        equirect, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    image[~disc] = 0
    return image


def _center_forward(equirect: np.ndarray) -> np.ndarray:
    """Put the left-lens axis at the middle of the frame.

    A 360 player opens looking at the center column. That column is then
    straight ahead, and dragging the view rotates around the panorama.
    """
    return np.roll(equirect, equirect.shape[1] // 4, axis=1)


def equirect_to_rectilinear(
    equirect: np.ndarray,
    out_width: int,
    out_height: int,
    fov_deg: float = 90.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Perspective view of an equirectangular panorama."""
    height, width = equirect.shape[:2]
    fov_deg = min(float(fov_deg), 150.0)
    half_h = math.radians(fov_deg / 2.0)
    half_v = half_h * out_height / max(out_width, 1)
    vs, us = np.mgrid[0:out_height, 0:out_width]
    u_norm = (us - (out_width - 1) / 2.0) / ((out_width - 1) / 2.0)
    v_norm = (vs - (out_height - 1) / 2.0) / ((out_height - 1) / 2.0)
    ray_x = u_norm * math.tan(half_h)
    ray_y = v_norm * math.tan(half_v)
    ray_z = np.ones_like(ray_x)
    ray_len = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    ray_x, ray_y, ray_z = ray_x / ray_len, ray_y / ray_len, ray_z / ray_len

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    ray_x, ray_z = ray_x * cos_yaw + ray_z * sin_yaw, -ray_x * sin_yaw + ray_z * cos_yaw
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    ray_y, ray_z = ray_y * cos_pitch - ray_z * sin_pitch, ray_y * sin_pitch + ray_z * cos_pitch

    lon = np.arctan2(ray_x, ray_z)
    lat = np.arcsin(np.clip(-ray_y, -1.0, 1.0))
    map_x = ((lon + math.pi) / (2.0 * math.pi) * width).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * height).astype(np.float32)
    return cv2.remap(
        equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )


def _load_calibration(calibration):
    """Accept a CameraCalibration or a path to a TOML or JSON config."""
    if calibration is None:
        return None
    if hasattr(calibration, "lens1") and hasattr(calibration, "lens2"):
        return calibration
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from camera_calibration.calib.calibration_config import CameraCalibration

    path = Path(calibration)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration not found: {path}")
    return CameraCalibration.load(path)


def _stitch_calibrated(frame: np.ndarray, calibration, base_fov_deg: float) -> np.ndarray:
    """Project one dual-fisheye frame with a fixed calibration.

    The lens model does not change between frames, so the seam stays put.
    """
    from camera_calibration.projections.fisheye_to_equirectangular import (
        mask_fisheye_circle,
        stitch_dual_fisheye,
    )

    height, width = frame.shape[:2]
    half = width // 2
    out_w = width if width % 2 == 0 else width - 1
    out_h = out_w // 2
    left, _left_mask = mask_fisheye_circle(frame[:, :half])
    right, _right_mask = mask_fisheye_circle(frame[:, half : half * 2])
    equirect, _left, _right, _overlap = stitch_dual_fisheye(
        left, right, calibration, base_fov_deg, out_w, out_h, True
    )
    if equirect.shape[0] != out_h or equirect.shape[1] != out_w:
        equirect = cv2.resize(equirect, (out_w, out_h))
    return _center_forward(equirect)


def render_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    output_format: str = "fisheye",
    calibration=None,
    base_fov_deg: float = 180.0,
    view_fov_deg: float = 90.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    lens_fov_deg: float = 200.0,
) -> str:
    """Write a video and return its path.

    ``output_format`` is ``fisheye`` (the original side-by-side image),
    ``rectalinear`` (a flat view of the calibrated stitch), or ``360``
    (equirectangular, tagged so a 360 player can look around).

    Rectilinear and 360 use ``calibration`` for every frame. Pass a
    ``calibration.toml`` path or a ``CameraCalibration``. ``lens_fov_deg`` is
    unused and kept so older callers still run.
    """
    del lens_fov_deg
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    fmt = _normalize_format(output_format)
    calib = _load_calibration(calibration)
    if fmt in ("360", "rectalinear") and calib is None:
        raise ValueError("360 and rectalinear need a calibration file (--calibration)")
    destination = Path(output_path) if output_path else source.with_name(f"{source.stem}_{fmt}.mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 2 or height < 2:
        capture.release()
        raise RuntimeError(f"Video has no frames: {source}")

    if fmt == "360":
        out_w = width if width % 2 == 0 else width - 1
        out_h = out_w // 2
    else:
        out_w = width if width % 2 == 0 else width - 1
        out_h = height if height % 2 == 0 else height - 1

    writer = _open_writer(destination, fps, out_w, out_h)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if fmt == "fisheye":
                written = frame if frame.shape[1] == out_w and frame.shape[0] == out_h else cv2.resize(frame, (out_w, out_h))
            else:
                equirect = _stitch_calibrated(frame, calib, base_fov_deg)
                if fmt == "360":
                    written = equirect if equirect.shape[1] == out_w else cv2.resize(equirect, (out_w, out_h))
                else:
                    written = equirect_to_rectilinear(
                        equirect, out_w, out_h, view_fov_deg, yaw_deg, pitch_deg
                    )
            writer.write(written)
    finally:
        writer.release()
        capture.release()

    if fmt == "360":
        _tag_equirectangular(destination)
    return str(destination)


def _normalize_format(output_format: str | None) -> str:
    name = (output_format or "fisheye").strip().lower()
    if name in ("", "fisheye", "original"):
        return "fisheye"
    if name in ("rectalinear", "rectilinear"):
        return "rectalinear"
    if name in ("360", "equirectangular", "equirect"):
        return "360"
    raise ValueError("output_format must be fisheye, rectalinear, or 360")


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for fourcc in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot create video writer for {path}")


_SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
_SPHERICAL_XML = (
    b'<?xml version="1.0"?>'
    b'<rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
    b"<GSpherical:Spherical>true</GSpherical:Spherical>"
    b"<GSpherical:Stitched>true</GSpherical:Stitched>"
    b"<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>"
    b"</rdf:SphericalVideo>"
)


def _tag_equirectangular(path: Path) -> None:
    """Mark an MP4 as a spherical equirectangular video, when the layout allows it.

    Players that understand the spherical tag open the file as a 360 view and
    let the user drag to look around. A plain equirectangular file is kept if
    the container cannot be tagged.
    """
    data = bytearray(path.read_bytes())
    if len(data) < 16 or data[4:8] not in (b"ftyp", b"mdat", b"moov", b"free"):
        return
    boxes = _top_boxes(data)
    moov = next((box for box in boxes if box[2] == b"moov"), None)
    if moov is None:
        return
    moov_pos, moov_size, _typ = moov
    trak_pos = _video_trak(data, moov_pos + 8, moov_pos + moov_size)
    if trak_pos is None:
        return
    trak_size = int.from_bytes(data[trak_pos : trak_pos + 4], "big")
    if trak_size < 8:
        return
    payload = _SPHERICAL_UUID + _SPHERICAL_XML
    uuid_box = (8 + len(payload)).to_bytes(4, "big") + b"uuid" + payload
    insert_at = trak_pos + trak_size
    data[insert_at:insert_at] = uuid_box
    added = len(uuid_box)
    _grow_box(data, trak_pos, added)
    _grow_box(data, moov_pos, added)
    moov_size_now = int.from_bytes(data[moov_pos : moov_pos + 4], "big")
    if any(start < insert_at and kind == b"mdat" for start, _size, kind in boxes):
        _shift_chunk_offsets(data, moov_pos + 8, moov_pos + moov_size_now, added, insert_at)
    path.write_bytes(data)


def _top_boxes(data: bytearray) -> list[tuple[int, int, bytes]]:
    boxes = []
    pos = 0
    end = len(data)
    while pos + 8 <= end:
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = bytes(data[pos + 4 : pos + 8])
        if size == 1 and pos + 16 <= end:
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
        elif size == 0:
            size = end - pos
        if size < 8 or pos + size > end:
            break
        boxes.append((pos, size, kind))
        pos += size
    return boxes


def _video_trak(data: bytearray, start: int, end: int) -> int | None:
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = bytes(data[pos + 4 : pos + 8])
        if size < 8 or pos + size > end:
            return None
        if kind == b"trak" and b"vide" in data[pos : pos + size]:
            return pos
        pos += size
    return None


def _grow_box(data: bytearray, pos: int, added: int) -> None:
    size = int.from_bytes(data[pos : pos + 4], "big")
    data[pos : pos + 4] = (size + added).to_bytes(4, "big")


_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"mvex"}


def _shift_chunk_offsets(data: bytearray, start: int, end: int, added: int, insert_at: int) -> None:
    """Keep sample locations valid when bytes were inserted before mdat."""
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = bytes(data[pos + 4 : pos + 8])
        if size < 8 or pos + size > end:
            return
        if kind in (b"stco", b"co64"):
            count = int.from_bytes(data[pos + 12 : pos + 16], "big")
            entry = 4 if kind == b"stco" else 8
            base = pos + 16
            for index in range(count):
                at = base + index * entry
                if at + entry > pos + size:
                    break
                offset = int.from_bytes(data[at : at + entry], "big")
                if offset >= insert_at:
                    data[at : at + entry] = (offset + added).to_bytes(entry, "big")
        elif kind in _MP4_CONTAINERS:
            _shift_chunk_offsets(data, pos + 8, pos + size, added, insert_at)
        pos += size


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        description="Write a fisheye, rectilinear, or rotatable 360 video."
    )
    parser.add_argument("video", help="Side-by-side dual-fisheye video.")
    parser.add_argument("-o", "--output", help="Output video path. Default: <name>_<format>.mp4")
    parser.add_argument(
        "--output_format",
        default="fisheye",
        help="fisheye (original, the default), rectalinear, or 360.",
    )
    parser.add_argument(
        "--calibration", "-c",
        help="Calibration TOML or JSON. Required for rectalinear and 360.",
    )
    parser.add_argument(
        "--base-fov", type=float, default=180.0,
        help="Longitude span of each lens in the stitch, in degrees (default: 180).",
    )
    parser.add_argument("--view-fov", type=float, default=90.0, help="Rectilinear view field of view in degrees.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rectilinear look yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rectilinear look pitch in degrees.")
    args = parser.parse_args(argv)
    path = render_video(
        args.video,
        args.output,
        args.output_format,
        calibration=args.calibration,
        base_fov_deg=args.base_fov,
        view_fov_deg=args.view_fov,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
    )
    print(path)
    return path


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
