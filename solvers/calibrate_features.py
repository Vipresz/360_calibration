#!/usr/bin/env python3
"""Feature-based calibration for dual-fisheye 360 cameras.

Pipeline
--------
1. Project each fisheye to its own equirectangular panorama (lens 1 = front
   hemisphere, lens 2 = back hemisphere) using the NOMINAL calibration.
2. Run SIFT or ORB on each panorama, restricted to the seam overlap zone
   (where both lenses see the world).
3. Match descriptors with a Lowe ratio test plus a latitude-consistency
   filter (matched scene points must have the same latitude).
4. RANSAC by world-ray residual: inliers are matches whose two world rays
   agree to within an angular threshold under the NOMINAL calibration.
5. Back-project each surviving match to its source FISHEYE pixel pair
   (using nominal calibration, so the fisheye pixels are fixed inputs).
6. scipy.optimize.least_squares (Levenberg-Marquardt / TRF) on every lens
   parameter, minimising the world-ray residual between every matched
   (fisheye_pixel_1, fisheye_pixel_2) pair.

The output is a CameraCalibration object identical in shape to the one
produced by calibrate_adjoint.py / calibrate_stepwise.py, so any caller
(create_calibration.py, the C++ viewer, etc.) is unaffected.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

try:
    from camera_calibration.calib.calibration_config import (
        CameraCalibration, LensCalibration,
    )
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, stitch_dual_fisheye,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import (
        CameraCalibration, LensCalibration,
    )
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, stitch_dual_fisheye,
    )


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def _make_detector(name: str):
    """Return (detector, norm).  ``norm`` is the BFMatcher norm to use.

    The seam-overlap strips are typically in the heavily-vignetted rim of
    each fisheye, so we relax the contrast/edge thresholds compared to
    SIFT defaults; otherwise SIFT routinely finds < 10 keypoints per strip.
    """
    name = name.lower()
    if name == 'sift':
        return cv2.SIFT_create(nfeatures=8000,
                               contrastThreshold=0.01,
                               edgeThreshold=20), cv2.NORM_L2
    if name == 'orb':
        return cv2.ORB_create(nfeatures=8000,
                              scaleFactor=1.2,
                              nlevels=8,
                              fastThreshold=8), cv2.NORM_HAMMING
    if name == 'akaze':
        return cv2.AKAZE_create(threshold=0.0005), cv2.NORM_HAMMING
    raise ValueError(
        f"Unknown detector '{name}'.  Valid: sift, orb, akaze."
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Same R = Rz @ Ry @ Rx convention as fisheye_to_equirect_calibrated."""
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def fisheye_pixels_to_world_rays(uv: np.ndarray, lens: LensCalibration,
                                  w: int, h: int, *, flip: bool = False) -> np.ndarray:
    """Vectorised forward camera model: (u, v) -> unit world ray (X, Y, Z).

    Inverts the projection that fisheye_to_equirect_calibrated implements.

    For lens 2 (the back lens) set ``flip=True``: the fliplr workflow used
    by stitch_dual_fisheye is applied here implicitly so the returned world
    ray is in the SHARED world frame (lens 1 boresight = +X, lens 2
    boresight = -X).
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    if flip:
        u = (w - 1) - u

    cx0 = lens.center_x * w
    cy0 = lens.center_y * h
    radius = min(w, h) / 2.0

    dx = u - cx0
    dy = v - cy0
    r_dist = np.sqrt(dx * dx + dy * dy) / radius
    theta  = np.arctan2(-dy, dx)

    # Inverse of r_dist = r * (1 + k1*r^2 + k2*r^4 + k3*r^6).  Fixed-point
    # iteration is fine for the small distortions we deal with.
    r = r_dist.copy()
    if abs(lens.k1) > 1e-9 or abs(lens.k2) > 1e-9 or abs(lens.k3) > 1e-9:
        for _ in range(10):
            factor = 1.0 + lens.k1 * r ** 2 + lens.k2 * r ** 4 + lens.k3 * r ** 6
            factor = np.where(np.abs(factor) < 1e-9, 1e-9, factor)
            r = r_dist / factor

    phi = r * math.radians(lens.fov / 2.0)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # (Px, Py, Pz) are the projection's intermediate "lens-frame" axes.
    # The forward map relates them to the rotated lens-world (Xlens, Ylens,
    # Zlens) by:  Px = -Zlens, Py = Ylens, Pz = Xlens.  Inverting:
    Px = sphi * np.cos(theta + math.pi / 2.0)
    Py = sphi * np.sin(theta + math.pi / 2.0)
    Pz = cphi
    Xlens =  Pz
    Ylens =  Py
    Zlens = -Px

    # The forward applies M (Rz Ry Rx) to take world -> lens.  So
    # world = M^T @ lens.  As row vectors: world_row = lens_row @ M.
    M = _rotation_matrix(lens.rotation_yaw, lens.rotation_pitch, lens.rotation_roll)
    P_lens  = np.stack([Xlens, Ylens, Zlens], axis=-1)
    P_world = P_lens @ M
    Xw = P_world[..., 0]
    Yw = P_world[..., 1]
    Zw = P_world[..., 2]

    if flip:
        # Lens 2 fliplr-workflow places its hemisphere in the BACK of the
        # shared canvas.  Per derivation: (X,Y,Z)_intermediate -> (-X, Y, Z).
        Xw = -Xw

    rays = np.stack([Xw, Yw, Zw], axis=-1)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return rays / norm


def equirect_pixels_to_fisheye_pixels(eq_uv: np.ndarray, lens: LensCalibration,
                                       w: int, h: int,
                                       proj_w: int, proj_h: int,
                                       canvas_fov: float,
                                       *, flip_canvas: bool = False) -> np.ndarray:
    """Vectorised inverse of fisheye_to_equirect_calibrated for given output
    pixels.  ``flip_canvas`` is True for lens 2's panorama (which was built
    with a fliplr step) so the equirect coords are pre-mirrored before back-
    projecting.
    """
    eq_uv = np.asarray(eq_uv, dtype=np.float64)
    if eq_uv.ndim == 1:
        eq_uv = eq_uv.reshape(1, 2)
    eu = eq_uv[:, 0]
    ev = eq_uv[:, 1]
    if flip_canvas:
        eu = (proj_w - 1) - eu

    lon_range = math.radians(canvas_fov)
    longitude = (eu / proj_w) * lon_range - lon_range / 2.0
    latitude  = (0.5 - ev / proj_h) * math.pi

    Xw = np.cos(latitude) * np.cos(longitude)
    Yw = np.cos(latitude) * np.sin(longitude)
    Zw = np.sin(latitude)

    M = _rotation_matrix(lens.rotation_yaw, lens.rotation_pitch, lens.rotation_roll)
    world = np.stack([Xw, Yw, Zw], axis=-1)
    rotated = world @ M.T

    Px = -rotated[..., 2]
    Py =  rotated[..., 1]
    Pz =  rotated[..., 0]

    phi   = np.arccos(np.clip(Pz, -1.0, 1.0))
    theta = np.arctan2(Py, Px) - math.pi / 2.0
    r     = phi / math.radians(lens.fov / 2.0)
    r_dist = r * (1.0 + lens.k1 * r ** 2 + lens.k2 * r ** 4 + lens.k3 * r ** 6)

    cx0 = lens.center_x * w
    cy0 = lens.center_y * h
    radius = min(w, h) / 2.0
    u = cx0 + r_dist * np.cos(theta) * radius
    v = cy0 - r_dist * np.sin(theta) * radius
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Feature detection + matching
# ---------------------------------------------------------------------------

def _project_lens(img: np.ndarray, lens: LensCalibration, *, flip: bool,
                   proj_w: int, proj_h: int, canvas_fov: float):
    """Project an input fisheye to its (own-hemisphere) equirect canvas.
    Mirrors stitch_dual_fisheye for lens 2 (flipped input then flipped output).
    """
    if flip:
        img_in = np.fliplr(img)
        patch, mask = fisheye_to_equirect_calibrated(
            img_in, proj_w, proj_h, lens, canvas_fov,
            edge_margin_ratio=0.0)
        return np.fliplr(patch), np.fliplr(mask)
    return fisheye_to_equirect_calibrated(
        img, proj_w, proj_h, lens, canvas_fov, edge_margin_ratio=0.0)


def _strip_col_range(canvas_fov: float, proj_w: int,
                      overlap_deg: float = 0.0,
                      overlap_center_deg: float = 0.0,
                      base_fov: float = 180.0) -> Tuple[int, int]:
    """Return (col_low, col_high) defining the right-edge seam strip in
    column indices, where the LEFT strip is the symmetric mirror.

    Mirrors adjoint_tuning.toml's seam_overlap_* convention:
      * ``overlap_deg <= 0``: legacy geometric overlap = ``lens_fov - base_fov``.
      * ``overlap_deg  > 0``: strip FOV [center - w/2, center + w/2], with
        ``center = base_fov`` when ``overlap_center_deg <= 0``.

    The strip is clamped to the canvas so [center +/- w/2] outside the lens
    coverage doesn't produce out-of-bounds masks.
    """
    if overlap_deg > 0.0:
        center_fov = overlap_center_deg if overlap_center_deg > 0.0 else base_fov
        strip_low_fov  = center_fov - overlap_deg / 2.0
        strip_high_fov = center_fov + overlap_deg / 2.0
    else:
        strip_low_fov  = base_fov
        strip_high_fov = canvas_fov  # = lens fov in our convention

    # FOV is the full angle (twice the lon).  Convert to longitudes (per side).
    strip_low_lon  = strip_low_fov  / 2.0
    strip_high_lon = strip_high_fov / 2.0

    # Clamp to the actual canvas coverage.
    canvas_half = canvas_fov / 2.0
    strip_low_lon  = max(0.0, min(strip_low_lon,  canvas_half))
    strip_high_lon = max(0.0, min(strip_high_lon, canvas_half))
    if strip_high_lon <= strip_low_lon:
        return -1, -1  # empty

    # col(lon) = (lon + canvas_fov/2) * proj_w / canvas_fov (right side, lon > 0)
    col_low  = int(math.floor((strip_low_lon  + canvas_half) / canvas_fov * proj_w))
    col_high = int(math.ceil ((strip_high_lon + canvas_half) / canvas_fov * proj_w))
    col_low  = max(0, min(col_low,  proj_w))
    col_high = max(col_low + 1, min(col_high, proj_w))
    return col_low, col_high


def _match_features_in_overlap(left_patch: np.ndarray, left_mask: np.ndarray,
                                right_patch: np.ndarray, right_mask: np.ndarray,
                                detector, norm: int,
                                canvas_fov: float,
                                proj_w: int,
                                ratio: float = 0.75,
                                lat_tol_px: int = 8,
                                overlap_deg: float = 0.0,
                                overlap_center_deg: float = 0.0,
                                base_fov: float = 180.0,
                                ) -> List[Tuple[Tuple[float, float],
                                                Tuple[float, float],
                                                float]]:
    """Detect features only inside the seam-overlap strips of each panorama
    and match strip-pair-vs-strip-pair (front-vs-front, back-vs-back).

    Without this restriction SIFT/ORB find thousands of features all over
    each panorama, and a Lowe ratio test on cross-strip pairs (e.g. lens 1
    front-seam vs lens 2 back-seam) discards almost everything because
    similar texture in unrelated parts of the scene poisons the ratio.

    ``overlap_deg`` / ``overlap_center_deg`` follow the same convention as
    adjoint_tuning.toml's ``seam_overlap_*`` and let you override the strip's
    width/centre in degrees of FOV.  Default 0/0 = use the geometric overlap.
    """
    col_low_right, col_high_right = _strip_col_range(
        canvas_fov, proj_w,
        overlap_deg=overlap_deg,
        overlap_center_deg=overlap_center_deg,
        base_fov=base_fov,
    )
    if col_low_right < 0 or col_high_right - col_low_right < 4:
        return []
    # Strip width on each side, in pixels.
    strip_w_px = col_high_right - col_low_right

    h, w = left_patch.shape[:2]

    def _gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    g1 = _gray(left_patch)
    g2 = _gray(right_patch)
    # Boost local contrast in the (often heavily vignetted) seam strips so
    # the detector finds enough keypoints to match.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g1 = clahe.apply(g1)
    g2 = clahe.apply(g2)

    left_mask_u8  = (left_mask  > 0).astype(np.uint8) * 255
    right_mask_u8 = (right_mask > 0).astype(np.uint8) * 255

    # By symmetry, the LEFT strip lives in the mirror columns of the RIGHT
    # strip.  E.g. RIGHT [1214, 1247] -> LEFT [proj_w - 1247, proj_w - 1214]
    # = [33, 66].
    col_low_left  = proj_w - col_high_right
    col_high_left = proj_w - col_low_right

    def _strip_mask(panorama_mask, *, side: str) -> np.ndarray:
        """``side`` = 'left' or 'right' (column edge of the panorama)."""
        out = np.zeros_like(panorama_mask)
        if side == 'left':
            out[:, col_low_left:col_high_left] = \
                panorama_mask[:, col_low_left:col_high_left]
        else:
            out[:, col_low_right:col_high_right] = \
                panorama_mask[:, col_low_right:col_high_right]
        return out

    # ---- Identify which strip pairs share world content ---------------------
    # Lens 1 panorama: lens 1 frame == world frame (boresight at lon=0).
    #   - cols [0, overlap_px)    : world lon in [-canvas/2, -180+canvas/2) (back seam)
    #   - cols [-overlap_px, end) : world lon in [180-canvas/2, canvas/2)   (front seam)
    # Lens 2 panorama (post-fliplr): boresight at world lon=pi.
    #   - cols [0, overlap_px)    : world lon in [180-canvas/2, ...) (FRONT seam)
    #   - cols [-overlap_px, end) : world lon in (-180+canvas/2, ...) (BACK seam)
    # So:  lens1.right_strip  matches lens2.left_strip  (front seam)
    #      lens1.left_strip   matches lens2.right_strip (back seam)
    pairs = [
        ('right', 'left',   'front_seam'),
        ('left',  'right',  'back_seam'),
    ]

    matcher = cv2.BFMatcher(norm)
    out: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    for side1, side2, label in pairs:
        m1 = _strip_mask(left_mask_u8,  side=side1)
        m2 = _strip_mask(right_mask_u8, side=side2)
        if int(np.sum(m1 > 0)) < 200 or int(np.sum(m2 > 0)) < 200:
            continue

        kp1, des1 = detector.detectAndCompute(g1, m1)
        kp2, des2 = detector.detectAndCompute(g2, m2)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        raw = matcher.knnMatch(des1, des2, k=2)
        kept = 0
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= ratio * n.distance:
                continue
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            # Latitude consistency: same world point => same equirect row.
            if abs(p1[1] - p2[1]) > lat_tol_px:
                continue
            out.append((p1, p2, float(m.distance)))
            kept += 1
    return out


def _ransac_filter_by_world_ray(matches: List[dict], angle_thresh_deg: float,
                                 ) -> List[dict]:
    """Drop matches whose two world rays disagree by more than ``angle_thresh``
    under the NOMINAL calibration.  This is a cheap geometric inlier test --
    it does not need a model to fit."""
    if not matches:
        return matches
    rays1 = np.stack([m['ray1'] for m in matches])
    rays2 = np.stack([m['ray2'] for m in matches])
    cosang = np.clip(np.sum(rays1 * rays2, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    keep = ang <= angle_thresh_deg
    return [m for m, k in zip(matches, keep) if k]


# ---------------------------------------------------------------------------
# Top-level: collect matches across all frames
# ---------------------------------------------------------------------------

def collect_matches(frames_data: Sequence[Tuple[np.ndarray, np.ndarray]],
                     init_lens1: LensCalibration,
                     init_lens2: LensCalibration,
                     *,
                     detector_name: str = 'sift',
                     ratio: float = 0.75,
                     ransac_angle_deg: float = 2.0,
                     overlap_deg: float = 0.0,
                     overlap_center_deg: float = 0.0,
                     base_fov: float = 180.0,
                     verbose: bool = True) -> List[dict]:
    """Run feature detection + matching over every frame; return a flat list
    of inlier matches.

    Each element is a dict with keys:
      'frame'        : frame index
      'fisheye1_uv'  : (u, v) source pixel in left_img
      'fisheye2_uv'  : (u, v) source pixel in right_img (ORIGINAL, not flipped)
      'ray1'         : initial world ray from lens 1
      'ray2'         : initial world ray from lens 2 (post-flip workflow)
      'descriptor_distance'
    """
    if not frames_data:
        return []

    detector, norm = _make_detector(detector_name)

    # Common projection size: roughly the input fisheye size, full FOV canvas.
    h, w = frames_data[0][0].shape[:2]
    proj_h = h
    proj_w = w * 2 if w < 800 else w
    canvas_fov = max(init_lens1.fov, init_lens2.fov)

    all_matches: List[dict] = []
    for fi, (left, right) in enumerate(frames_data):
        left_patch,  left_mask  = _project_lens(left,  init_lens1, flip=False,
                                                 proj_w=proj_w, proj_h=proj_h,
                                                 canvas_fov=canvas_fov)
        right_patch, right_mask = _project_lens(right, init_lens2, flip=True,
                                                 proj_w=proj_w, proj_h=proj_h,
                                                 canvas_fov=canvas_fov)

        m = _match_features_in_overlap(
            left_patch, left_mask, right_patch, right_mask,
            detector, norm,
            canvas_fov=canvas_fov, proj_w=proj_w,
            ratio=ratio,
            lat_tol_px=max(8, proj_h // 80),
            overlap_deg=overlap_deg,
            overlap_center_deg=overlap_center_deg,
            base_fov=base_fov)
        if not m:
            if verbose:
                print(f"  frame {fi}: 0 raw matches")
            continue

        # Back-project equirect coords to FISHEYE source pixels using the
        # nominal calibration.  These pixel pairs are the data points we'll
        # pass to least_squares.
        left_eq_uv  = np.array([p[0] for p in m], dtype=np.float64)
        right_eq_uv = np.array([p[1] for p in m], dtype=np.float64)

        fish1 = equirect_pixels_to_fisheye_pixels(
            left_eq_uv,  init_lens1, w, h, proj_w, proj_h,
            canvas_fov, flip_canvas=False)
        fish2 = equirect_pixels_to_fisheye_pixels(
            right_eq_uv, init_lens2, w, h, proj_w, proj_h,
            canvas_fov, flip_canvas=True)
        # fish2 is in the FLIPPED frame; convert back to original right_img
        # coordinates.
        fish2[:, 0] = (w - 1) - fish2[:, 0]

        # Drop fisheye pixels that fell outside the input image rectangle.
        ok1 = (fish1[:, 0] >= 0) & (fish1[:, 0] <= w - 1) & \
              (fish1[:, 1] >= 0) & (fish1[:, 1] <= h - 1)
        ok2 = (fish2[:, 0] >= 0) & (fish2[:, 0] <= w - 1) & \
              (fish2[:, 1] >= 0) & (fish2[:, 1] <= h - 1)
        ok  = ok1 & ok2
        fish1 = fish1[ok]
        fish2 = fish2[ok]
        descriptor_distances = [d for d, k in zip([p[2] for p in m], ok) if k]
        if len(fish1) == 0:
            continue

        rays1 = fisheye_pixels_to_world_rays(fish1, init_lens1, w, h, flip=False)
        rays2 = fisheye_pixels_to_world_rays(fish2, init_lens2, w, h, flip=True)

        for i in range(len(fish1)):
            all_matches.append({
                'frame':                fi,
                'fisheye1_uv':          fish1[i],
                'fisheye2_uv':          fish2[i],
                'ray1':                 rays1[i],
                'ray2':                 rays2[i],
                'descriptor_distance':  descriptor_distances[i],
            })

        if verbose:
            print(f"  frame {fi}: {len(m):4d} raw matches -> {len(fish1):4d} on-image")

    if verbose:
        print(f"Pre-RANSAC matches: {len(all_matches)}")

    # RANSAC by initial-ray agreement.
    inliers = _ransac_filter_by_world_ray(all_matches, ransac_angle_deg)
    if verbose:
        print(f"RANSAC kept     : {len(inliers)} (angle threshold "
              f"{ransac_angle_deg:.2f} deg)")
    return inliers


# ---------------------------------------------------------------------------
# Optimisation: parameter packing + residuals
# ---------------------------------------------------------------------------

# Parameter vector layout: 9 per lens, 18 total.
_LENS_PARAM_KEYS = (
    'center_x', 'center_y', 'fov',
    'k1', 'k2', 'k3',
    'rotation_yaw', 'rotation_pitch', 'rotation_roll',
)


def _lens_to_vec(lens: LensCalibration) -> np.ndarray:
    return np.array([getattr(lens, k) for k in _LENS_PARAM_KEYS], dtype=np.float64)


def _vec_to_lens(vec: np.ndarray) -> LensCalibration:
    return LensCalibration(**{k: float(vec[i]) for i, k in enumerate(_LENS_PARAM_KEYS)})


# Gentle Tikhonov pull toward the initial guess, applied as additional
# residual rows in the least_squares objective.  The seam-only feature data
# is under-determined for the full 9-DOF-per-lens space (latitudes all near
# the equator), so without this pull the optimizer happily trades large
# k1/k2/k3 deltas for tiny gains in seam accuracy.  Weights are in the same
# scale as a 3-D ray residual (range ~ [0, 2]); a value of 0.02 means the
# regulariser equals one match's worth of residual when a parameter drifts
# by `1 / weight` units from its initial value.
_DEFAULT_REG_WEIGHTS = np.array([
    # center_x, center_y, fov,    k1,    k2,    k3,    yaw,   pitch, roll
    20.0,    20.0,    0.05,  3.0,   3.0,   10.0,  3.0,   3.0,   3.0,
], dtype=np.float64)


def _residuals(x: np.ndarray, fisheye1_uv: np.ndarray, fisheye2_uv: np.ndarray,
                w: int, h: int,
                x0: Optional[np.ndarray] = None,
                reg_weights: Optional[np.ndarray] = None) -> np.ndarray:
    lens1 = _vec_to_lens(x[:9])
    lens2 = _vec_to_lens(x[9:])
    rays1 = fisheye_pixels_to_world_rays(fisheye1_uv, lens1, w, h, flip=False)
    rays2 = fisheye_pixels_to_world_rays(fisheye2_uv, lens2, w, h, flip=True)
    data_res = (rays1 - rays2).reshape(-1)
    if x0 is None:
        return data_res
    if reg_weights is None:
        reg_weights = _DEFAULT_REG_WEIGHTS
    weights = np.tile(reg_weights, 2)
    reg_res = (x - x0) * weights
    return np.concatenate([data_res, reg_res])


def _build_bounds(init_lens1: LensCalibration, init_lens2: LensCalibration,
                   *, allow_rotation: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Reasonable built-in bounds for a Gear-360-class dual fisheye.

    Used as a fallback when no features_tuning.toml is supplied (see
    ``_config_to_init_and_bounds`` for the config-driven path).
    """
    def lens_bounds(lens):
        lo = np.array([
            lens.center_x - 0.04, lens.center_y - 0.04, max(160.0, lens.fov - 25.0),
            -0.15, -0.15, -0.05,
        ], dtype=np.float64)
        hi = np.array([
            lens.center_x + 0.04, lens.center_y + 0.04, min(220.0, lens.fov + 25.0),
             0.15,  0.15,  0.05,
        ], dtype=np.float64)
        if allow_rotation:
            lo = np.concatenate([lo, [-0.15, -0.15, -0.15]])
            hi = np.concatenate([hi, [ 0.15,  0.15,  0.15]])
        else:
            lo = np.concatenate([lo, [lens.rotation_yaw   - 1e-6,
                                       lens.rotation_pitch - 1e-6,
                                       lens.rotation_roll  - 1e-6]])
            hi = np.concatenate([hi, [lens.rotation_yaw   + 1e-6,
                                       lens.rotation_pitch + 1e-6,
                                       lens.rotation_roll  + 1e-6]])
        return lo, hi

    lo1, hi1 = lens_bounds(init_lens1)
    lo2, hi2 = lens_bounds(init_lens2)
    return np.concatenate([lo1, lo2]), np.concatenate([hi1, hi2])


# Names of fields in features_tuning.toml's lens section, in the same order
# as the 18-vector layout used by ``_lens_to_vec`` / ``_vec_to_lens``.
_TOML_PARAM_NAMES = (
    'center_x', 'center_y', 'fov',
    'k1', 'k2', 'k3',
    'rotation_yaw', 'rotation_pitch', 'rotation_roll',
)


def _reg_weights_from_config(config) -> np.ndarray:
    """Pull per-parameter Tikhonov weights from a ``[regularization]`` section."""
    if config is None or not getattr(config, 'features_regularization', None):
        return _DEFAULT_REG_WEIGHTS
    fr = config.features_regularization
    if not fr.get('enabled', True):
        return np.zeros_like(_DEFAULT_REG_WEIGHTS)
    keys = ('weight_center_x', 'weight_center_y', 'weight_fov',
            'weight_k1', 'weight_k2', 'weight_k3',
            'weight_rotation_yaw', 'weight_rotation_pitch', 'weight_rotation_roll')
    return np.array([float(fr[k]) for k in keys], dtype=np.float64)


def _config_to_init_and_bounds(config) -> Tuple[LensCalibration, LensCalibration,
                                                  np.ndarray, np.ndarray,
                                                  list]:
    """Translate a ``calibrate_adjoint.OptimizationConfig`` into:
        (init_lens1, init_lens2, lo, hi, tied_pairs)

    where ``tied_pairs`` is the list of (lens2_idx_in_18vec, lens1_idx) couples
    for parameters with ``use_lens1=true``.  These are enforced after every
    residual evaluation so the tied lens-2 parameter exactly mirrors lens 1.

    Pinned parameters (``optimize=false``) get bounds [nominal, nominal+eps]
    so least_squares treats them as constants.
    """
    EPS = 1e-9
    nom = {}
    lo = []
    hi = []
    for i_lens, lens_name in enumerate(('lens1', 'lens2')):
        for i_p, p_name in enumerate(_TOML_PARAM_NAMES):
            key = f"{lens_name}.{p_name}"
            spec = config.params[key]
            n = float(spec['nominal'])
            if spec.get('use_lens1') and not spec['optimize'] and lens_name == 'lens2':
                # Will be tied to lens1 after each eval; init to lens1's nominal.
                n = float(config.params[f'lens1.{p_name}']['nominal'])
            nom[(i_lens, i_p)] = n
            if spec['optimize']:
                lo.append(float(spec['min']))
                hi.append(float(spec['max']))
            else:
                lo.append(n - EPS)
                hi.append(n + EPS)

    def _lens_from_nom(i_lens):
        kwargs = {p: nom[(i_lens, j)] for j, p in enumerate(_TOML_PARAM_NAMES)}
        return LensCalibration(**kwargs)

    init_lens1 = _lens_from_nom(0)
    init_lens2 = _lens_from_nom(1)

    tied_pairs = []
    for i_p, p_name in enumerate(_TOML_PARAM_NAMES):
        spec2 = config.params[f'lens2.{p_name}']
        if spec2.get('use_lens1') and not spec2['optimize']:
            # lens2 idx = 9 + i_p, lens1 idx = i_p
            tied_pairs.append((9 + i_p, i_p))

    return init_lens1, init_lens2, np.asarray(lo), np.asarray(hi), tied_pairs


# ---------------------------------------------------------------------------
# Live preview during optimisation
# ---------------------------------------------------------------------------

def _make_preview(left_img: np.ndarray, right_img: np.ndarray,
                   lens1: LensCalibration, lens2: LensCalibration,
                   base_fov: float, *, max_w: int = 1280) -> np.ndarray:
    calib = CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)
    try:
        out, _, _, _ = stitch_dual_fisheye(left_img, right_img, calib, base_fov, blend=False)
    except Exception as exc:
        out = np.zeros_like(left_img)
        cv2.putText(out, f"projection error: {exc}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if out.shape[1] > max_w:
        s  = max_w / out.shape[1]
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calibrate_features(frames_data: Sequence[Tuple[np.ndarray, np.ndarray]],
                        base_fov: float,
                        *,
                        init_lens1: Optional[LensCalibration] = None,
                        init_lens2: Optional[LensCalibration] = None,
                        detector_name: Optional[str] = None,
                        ratio: Optional[float] = None,
                        ransac_angle_deg: Optional[float] = None,
                        allow_rotation: bool = True,
                        config=None,
                        max_iter: int = 200,
                        preview: bool = False,
                        preview_window: str = 'feature calibration preview',
                        verbose: bool = True) -> CameraCalibration:
    """Run the feature-based dual-fisheye calibration.

    Args:
        frames_data: iterable of (left_img, right_img) BGR pairs.
        base_fov:    the equirectangular base FOV in degrees (typically 180
                     -- see create_calibration.py --fov).  Used only for the
                     preview render; the residual itself is in 3-D world-ray
                     space and does not depend on it.
        init_lens1, init_lens2: starting lens calibration.  Ignored if
                     ``config`` is supplied (then nominals come from the TOML).
                     Defaults to a centred 195-deg lens.
        detector_name, ratio, ransac_angle_deg:
                     CLI-overridable knobs.  If ``None`` and ``config`` carries
                     a ``[features]`` section, the value comes from there;
                     otherwise built-in defaults apply.
        allow_rotation: if False, lens rotations are pinned at their initial
                     values.  Ignored when ``config`` is supplied (then the
                     per-rotation ``optimize`` flags in the TOML decide).
        config:      optional ``calibrate_adjoint.OptimizationConfig`` parsed
                     from a features_tuning.toml.  When supplied:
                       * lens1/lens2 nominal values become the init guess;
                       * ``min`` / ``max`` per parameter become the bounds;
                       * parameters with ``optimize=false`` are pinned;
                       * lens2 entries with ``use_lens1=true`` mirror lens1
                         exactly after every residual evaluation;
                       * the ``[features]`` section supplies detector / ratio
                         / RANSAC defaults;
                       * the ``[regularization]`` section supplies the per-
                         parameter Tikhonov weights.

    Returns: an optimised CameraCalibration.
    """
    # Pull defaults from the config's [features] section, then let any
    # explicit CLI argument override.
    feat_cfg = getattr(config, 'features', None) if config is not None else None
    feat_cfg = feat_cfg or {}
    if detector_name is None:
        detector_name = str(feat_cfg.get('detector', 'sift'))
    if ratio is None:
        ratio = float(feat_cfg.get('ratio', 0.85))
    if ransac_angle_deg is None:
        ransac_angle_deg = float(feat_cfg.get('ransac_deg', 5.0))
    MIN_MATCHES = int(feat_cfg.get('min_matches', 12))
    overlap_deg        = float(feat_cfg.get('overlap_deg',        0.0))
    overlap_center_deg = float(feat_cfg.get('overlap_center_deg', 0.0))

    reg_weights = _reg_weights_from_config(config)

    if config is not None:
        init_lens1, init_lens2, lo_cfg, hi_cfg, tied_pairs = \
            _config_to_init_and_bounds(config)
    else:
        if init_lens1 is None:
            init_lens1 = LensCalibration(0.5, 0.5, 195.0)
        if init_lens2 is None:
            init_lens2 = LensCalibration(0.5, 0.5, 195.0)
        lo_cfg = hi_cfg = None
        tied_pairs = []

    if not frames_data:
        raise ValueError("frames_data is empty")

    h, w = frames_data[0][0].shape[:2]

    print()
    print("=" * 60)
    print(f"FEATURE-BASED CALIBRATION ({detector_name.upper()})")
    print("=" * 60)
    print(f"  Frames        : {len(frames_data)}")
    print(f"  Frame size    : {w}x{h} per half")
    print(f"  Detector      : {detector_name}")
    print(f"  Ratio test    : {ratio}")
    print(f"  RANSAC tol    : {ransac_angle_deg} deg")
    print(f"  Min matches   : {MIN_MATCHES}")
    if overlap_deg > 0.0:
        ctr = overlap_center_deg if overlap_center_deg > 0.0 else 180.0
        print(f"  Strip (FOV)   : {overlap_deg:g} deg wide, "
              f"centred at {ctr:g} deg")
    else:
        print(f"  Strip (FOV)   : geometric (lens.fov - {base_fov:.0f} deg)")
    print(f"  Optimise rot. : {allow_rotation}")

    # Resolution warning: SIFT/ORB need real texture to find features in the
    # narrow seam annulus.  At 160 px per half the seam is ~20 px wide, far
    # too thin to land 30+ matches.
    if min(w, h) < 480:
        print()
        print(f"  WARNING: input is only {w}x{h} per half -- the seam annulus")
        print(f"           will be ~{min(w, h)//2 * (max(init_lens1.fov, init_lens2.fov) - base_fov) / max(init_lens1.fov, init_lens2.fov):.0f} px wide and SIFT/ORB may find very")
        print(f"           few matches.  Re-run with --scale 1.0 (default for")
        print(f"           the features solver) for best results.")

    # --- Stage 1: detect + match ---------------------------------------------
    # Try once at the user's requested settings; if that fails, automatically
    # retry with progressively looser thresholds so a near-correct nominal
    # calibration doesn't immediately blow up.
    _strip_kwargs = dict(overlap_deg=overlap_deg,
                         overlap_center_deg=overlap_center_deg,
                         base_fov=base_fov)
    matches = collect_matches(
        frames_data, init_lens1, init_lens2,
        detector_name=detector_name, ratio=ratio,
        ransac_angle_deg=ransac_angle_deg, verbose=verbose,
        **_strip_kwargs,
    )

    if len(matches) < MIN_MATCHES:
        # Fallback 1: relax RANSAC (initial calibration may be off by a few deg)
        print(f"\n  Only {len(matches)} matches at ransac={ransac_angle_deg} deg -- "
              f"retrying with ransac=15 deg ...")
        matches = collect_matches(
            frames_data, init_lens1, init_lens2,
            detector_name=detector_name, ratio=ratio,
            ransac_angle_deg=15.0, verbose=verbose,
            **_strip_kwargs,
        )

    if len(matches) < MIN_MATCHES:
        # Fallback 2: relax Lowe ratio (low-texture / low-resolution seam)
        print(f"\n  Only {len(matches)} matches -- retrying with ratio=0.92, ransac=20 deg ...")
        matches = collect_matches(
            frames_data, init_lens1, init_lens2,
            detector_name=detector_name, ratio=0.92,
            ransac_angle_deg=20.0, verbose=verbose,
            **_strip_kwargs,
        )

    if len(matches) < MIN_MATCHES:
        raise RuntimeError(
            f"Only {len(matches)} feature matches survived even with relaxed "
            f"thresholds (need at least {MIN_MATCHES}).  Likely causes:\n"
            f"  - Scene has no texture in the seam region (uniform sky/wall) --\n"
            f"    re-shoot with the camera in a textured environment.\n"
            f"  - Input resolution too low: re-run with --scale 1.0 (the\n"
            f"    features solver does not need downsampling).\n"
            f"  - Initial calibration is off by more than ~20 deg -- run the\n"
            f"    adjoint or stepwise solver first to get a starting guess.\n"
            f"You can also try --detector orb or --detector akaze."
        )

    fisheye1 = np.stack([m['fisheye1_uv'] for m in matches])
    fisheye2 = np.stack([m['fisheye2_uv'] for m in matches])
    n_data = 3 * len(matches)

    # Initial residual norm (data part only).
    x0 = np.concatenate([_lens_to_vec(init_lens1), _lens_to_vec(init_lens2)])
    r0 = _residuals(x0, fisheye1, fisheye2, w, h)  # x0=None -> data only
    init_rms = float(np.sqrt(np.mean(r0 ** 2)))
    init_ang = float(np.degrees(2.0 * np.arcsin(np.clip(init_rms / 2.0, 0.0, 1.0))))
    print(f"  Initial mean angular error: {init_ang:.3f} deg "
          f"(ray RMS {init_rms:.5f})")

    # --- Stage 2: optimise ---------------------------------------------------
    if lo_cfg is not None:
        lo, hi = lo_cfg.copy(), hi_cfg.copy()
        tied_l2_idxs = {i2 for i2, _ in tied_pairs}
        if verbose:
            n_free = sum((hi - lo) > 1e-6)
            n_tied = len(tied_pairs)
            print(f"  Bounds source : features_tuning.toml "
                  f"({n_free} free, {n_tied} tied to lens1, "
                  f"{18 - n_free - n_tied} pinned)")
            for i, name in enumerate(_TOML_PARAM_NAMES + _TOML_PARAM_NAMES):
                lens = 'lens1' if i < 9 else 'lens2'
                free = (hi[i] - lo[i]) > 1e-6
                if free:
                    print(f"    {lens}.{name:14s}: [{lo[i]:.4f}, {hi[i]:.4f}]")
                elif i in tied_l2_idxs:
                    print(f"    {lens}.{name:14s}: tied to lens1.{name}")
                else:
                    print(f"    {lens}.{name:14s}: pinned at {lo[i]:.4f}")
    else:
        lo, hi = _build_bounds(init_lens1, init_lens2, allow_rotation=allow_rotation)

    # Make sure the initial point is inside the bounds.
    x0_clipped = np.clip(x0, lo, hi)
    # Apply ties up-front so the first eval is consistent.
    for i2, i1 in tied_pairs:
        x0_clipped[i2] = x0_clipped[i1]

    preview_window_open = False
    preview_first_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if preview:
        preview_first_pair = (frames_data[0][0].copy(), frames_data[0][1].copy())
        try:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window, 1024, 512)
            cv2.imshow(preview_window,
                       _make_preview(*preview_first_pair, init_lens1, init_lens2, base_fov))
            cv2.waitKey(1)
            preview_window_open = True
        except cv2.error:
            preview_window_open = False

    iter_count = [0]
    x0_for_reg = x0_clipped.copy()
    # Track best-so-far so we only log/preview when there's a real improvement.
    # least_squares burns ~18 evals per trust-region step doing finite-
    # difference Jacobians, none of which improve the cost; printing every
    # 5th eval makes that look like the optimiser is stuck.
    best_rms = [float('inf')]

    def _residuals_with_progress(x):
        # Honor "use_lens1=true" ties: lens2 entry mirrors lens1 in real time.
        if tied_pairs:
            x = x.copy()
            for i2, i1 in tied_pairs:
                x[i2] = x[i1]
        r = _residuals(x, fisheye1, fisheye2, w, h,
                        x0=x0_for_reg, reg_weights=reg_weights)
        iter_count[0] += 1

        rms = float(np.sqrt(np.mean(r[:n_data] ** 2)))
        # Print when this eval is materially better than anything we've seen
        # so far -- effectively once per accepted trust-region step.
        if rms < best_rms[0] * 0.999:
            best_rms[0] = rms
            ang = float(np.degrees(2.0 * np.arcsin(np.clip(rms / 2.0, 0.0, 1.0))))
            print(f"    step (eval {iter_count[0]:4d}): ray RMS={rms:.5f}  "
                  f"angular={ang:.4f} deg")
            if preview_window_open and preview_first_pair is not None:
                l1 = _vec_to_lens(x[:9])
                l2 = _vec_to_lens(x[9:])
                try:
                    cv2.imshow(
                        preview_window,
                        _make_preview(*preview_first_pair, l1, l2, base_fov))
                    cv2.waitKey(1)
                except cv2.error:
                    pass
        return r

    print(f"\n  Optimising 18 parameters over {len(matches)} matches "
          f"({len(matches) * 3} residuals) ...")
    # Why these knobs:
    #   x_scale='jac'  -- auto-scale the trust region by Jacobian column
    #                     norms.  Critical here because parameter ranges
    #                     differ by 250x (center_x = 0.04, fov = 10).
    #   tolerances 1e-6 / 1e-7 -- 1e-10 (the previous setting) wastes
    #                     thousands of finite-difference evals chasing
    #                     ray-RMS deltas of 1e-5 that are well below feature-
    #                     localisation noise.
    #   max_nfev capped at 600 -- with x_scale='jac' a healthy run converges
    #                     in 100-300 nfev; if it doesn't, more iterations
    #                     are unlikely to help (you're stuck in a flat valley).
    res = least_squares(
        _residuals_with_progress, x0_clipped,
        bounds=(lo, hi),
        method='trf',
        x_scale='jac',
        max_nfev=min(max_iter * 20, 600),
        xtol=1e-6, ftol=1e-7, gtol=1e-8,
        verbose=0,
    )

    final_data = res.fun[:n_data]
    final_rms = float(np.sqrt(np.mean(final_data ** 2)))
    final_ang = float(np.degrees(2.0 * np.arcsin(np.clip(final_rms / 2.0, 0.0, 1.0))))
    print(f"  Final  mean angular error: {final_ang:.3f} deg "
          f"(ray RMS {final_rms:.5f})")
    print(f"  scipy: {res.message}, {res.nfev} evals, status {res.status}")

    # Apply ties to the returned solution so the saved calibration matches
    # what the optimizer actually evaluated.
    x_final = res.x.copy()
    for i2, i1 in tied_pairs:
        x_final[i2] = x_final[i1]
    out_lens1 = _vec_to_lens(x_final[:9])
    out_lens2 = _vec_to_lens(x_final[9:])

    print()
    print("=" * 60)
    print("OPTIMIZED PARAMETERS")
    print("=" * 60)
    for i, lens in enumerate([out_lens1, out_lens2], 1):
        print(f"\nLens {i}:")
        print(f"  center: ({lens.center_x:.6f}, {lens.center_y:.6f})")
        print(f"  fov:    {lens.fov:.2f} deg")
        print(f"  k:      [{lens.k1:+.6f}, {lens.k2:+.6f}, {lens.k3:+.6f}]")
        print(f"  rot:    [{lens.rotation_yaw:+.6f}, "
              f"{lens.rotation_pitch:+.6f}, {lens.rotation_roll:+.6f}]")

    if preview_window_open and preview_first_pair is not None:
        try:
            cv2.imshow(
                preview_window,
                _make_preview(*preview_first_pair, out_lens1, out_lens2, base_fov))
            print("Press any key in the preview window to close it.")
            cv2.waitKey(0)
        except cv2.error:
            pass
        finally:
            try:
                cv2.destroyWindow(preview_window)
            except cv2.error:
                pass

    return CameraCalibration(lens1=out_lens1, lens2=out_lens2, is_horizontal=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description='Feature-based dual-fisheye calibration')
    p.add_argument('--image', '-i', help='Single dual-fisheye image (left | right)')
    p.add_argument('--video', help='Video; first frame is used by default')
    p.add_argument('--frames', type=int, default=5,
                   help='How many evenly-spaced video frames to use')
    p.add_argument('--scale', type=float, default=1.0,
                   help='Downsample input by this factor (default 1.0)')
    p.add_argument('--detector', choices=['sift', 'orb', 'akaze'], default='sift')
    p.add_argument('--ratio', type=float, default=0.75)
    p.add_argument('--ransac-deg', type=float, default=5.0)
    p.add_argument('--no-rotation', action='store_true',
                   help='Pin lens rotations at zero (only solve geometry).')
    p.add_argument('--fov', type=float, default=180.0,
                   help='Equirectangular base FOV (for preview render only).')
    p.add_argument('--preview', action='store_true')
    p.add_argument('-o', '--output', required=True,
                   help='Output calibration file. Extension chooses format: '
                        '.toml (default; viewer-compatible) or .json '
                        '(internal CameraCalibration format).')
    p.add_argument('--save-json', metavar='FILE',
                   help='Also write the raw CameraCalibration JSON to FILE '
                        '(in addition to --output).')
    args = p.parse_args()

    if not args.image and not args.video:
        p.error("provide --image or --video")

    frames_data: List[Tuple[np.ndarray, np.ndarray]] = []
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            p.error(f"cannot read {args.image}")
        h, w = img.shape[:2]
        frames_data = [(img[:, :w // 2], img[:, w // 2:])]
    else:
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            p.error("video reports zero frames")
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, min(args.frames, total))]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                hh, ww = fr.shape[:2]
                frames_data.append((fr[:, :ww // 2], fr[:, ww // 2:]))
        cap.release()

    if args.scale < 1.0:
        scaled: List[Tuple[np.ndarray, np.ndarray]] = []
        for l, r in frames_data:
            hh, ww = l.shape[:2]
            nh, nw = int(hh * args.scale), int(ww * args.scale)
            scaled.append((cv2.resize(l, (nw, nh), interpolation=cv2.INTER_AREA),
                           cv2.resize(r, (nw, nh), interpolation=cv2.INTER_AREA)))
        frames_data = scaled

    calib = calibrate_features(
        frames_data, args.fov,
        detector_name=args.detector,
        ratio=args.ratio,
        ransac_angle_deg=args.ransac_deg,
        allow_rotation=not args.no_rotation,
        preview=args.preview,
    )

    out_path = args.output
    out_lower = out_path.lower()
    if out_lower.endswith('.toml') or not (out_lower.endswith('.json')):
        # Default: TOML, viewer-compatible.  Reuse the writer from
        # create_calibration.py so the output is byte-identical to the other
        # solvers' output.
        try:
            from camera_calibration.calib.create_calibration import write_toml
        except ModuleNotFoundError:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from camera_calibration.calib.create_calibration import write_toml
        write_toml(calib.to_dict(), out_path, method='features_optimizer')
        print(f"\nTOML saved: {out_path}")
        print("Use with the Python viewer:")
        print(f"  gear360_viewer.py --calibration {out_path}")
        print("Copy to use with the C++ viewer:")
        print(f"  cpp/calibration.toml")
    else:
        calib.save_json(out_path)
        print(f"\nJSON saved: {out_path}")

    if args.save_json:
        calib.save_json(args.save_json)
        print(f"JSON saved: {args.save_json}")


if __name__ == '__main__':
    _cli()
