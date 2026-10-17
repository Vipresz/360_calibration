"""Synthetic checks for the real-time seam stitcher.

Accuracy is the fraction of overlap pixels whose gray value is within 8 levels
of the reference. The threshold is 99 percent. Smooth panoramas are used so
bilinear resampling is not counted as a seam error.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_calibration.realtime.stitcher import (  # noqa: E402
    RealtimeStitcher,
    render_fisheye_from_equirect,
)


FOV = 228.0
BASE = 180.0
LEVELS = 8.0
PROJ_H = 180
PROJ_W = 228
OVERLAP = PROJ_W - PROJ_H


def _panorama(height: int, width: int) -> np.ndarray:
    """Soft vertical bars plus a slow gradient.

    The bars give the seam search a unique peak. The edges are wide enough
    that cubic resampling stays inside an 8-level error.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.full((height, width, 3), 70, np.float32)
    for x0 in range(18, width, 36):
        bar = np.exp(-((xs - x0) ** 2) / (2 * 3.5**2))
        image[..., 0] += 150 * bar
        image[..., 1] += 90 * bar * (0.5 + 0.5 * np.sin(ys / 19.0))
        image[..., 2] += 40 * bar
    image[..., 1] += 18 * np.sin(2 * np.pi * ys / 90.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def _lenses(proj_h: int, proj_w: int, overlap: int, shift_x: float = 0.0, right_scale: float = 1.0):
    """Build a dual-fisheye frame whose overlap is a shifted, scaled copy.

    The right view starts ``proj_w - overlap`` columns into the canvas, so its
    left edge is the same scene as the left view's right edge.
    """
    canvas = _panorama(proj_h, proj_w * 2)
    left_view = canvas[:, :proj_w]
    origin = proj_w - overlap
    right_view = canvas[:, origin : origin + proj_w]
    if shift_x:
        right_view = np.roll(right_view, int(round(shift_x)), axis=1)
    size = proj_h
    left = render_fisheye_from_equirect(left_view, size, FOV)
    right = render_fisheye_from_equirect(right_view, size, FOV)
    if right_scale != 1.0:
        right = np.clip(right.astype(np.float32) * right_scale, 0, 255).astype(np.uint8)
    return np.hstack([left, right]), left_view


def _overlap_agreement(left_edge: np.ndarray, right_edge: np.ndarray) -> float:
    """Fraction of mid-latitude overlap pixels within ``LEVELS`` gray levels."""
    gray_a = left_edge.astype(np.float32).mean(axis=2)
    gray_b = right_edge.astype(np.float32).mean(axis=2)
    height = gray_a.shape[0]
    y0, y1 = int(height * 0.25), int(height * 0.75)
    gray_a = gray_a[y0:y1]
    gray_b = gray_b[y0:y1]
    valid = (gray_a > 2.0) & (gray_b > 2.0)
    if int(valid.sum()) < 50:
        return 0.0
    return float(np.mean(np.abs(gray_a[valid] - gray_b[valid]) <= LEVELS))


def _corrected_overlap(stitcher: RealtimeStitcher, frame: np.ndarray, estimate):
    """Reapply the published seam state and return the two overlap strips."""
    from camera_calibration.realtime.stitcher import _apply_gain, _warp

    left = frame[:, : frame.shape[1] // 2]
    right = frame[:, frame.shape[1] // 2 :]
    proj_w = int(round(PROJ_H * FOV / BASE))
    left_patch = stitcher._project(left, proj_w, PROJ_H)
    right_patch = np.fliplr(stitcher._project(np.fliplr(right), proj_w, PROJ_H))
    shifted = _warp(right_patch, estimate.shift_x, estimate.shift_y)
    _gain, row_gain = stitcher._measure_gain(left_patch, shifted, proj_w - PROJ_H)
    corrected = _apply_gain(shifted, estimate.gain, row_gain)
    overlap = proj_w - PROJ_H
    return left_patch[:, -overlap:], corrected[:, :overlap]


class RealtimeStitchTests(unittest.TestCase):
    def test_roundtrip_overlap_is_at_least_99_percent(self):
        stitcher = RealtimeStitcher(FOV, BASE, output_width=360, smoothing=0.0)
        frame, left_view = _lenses(PROJ_H, PROJ_W, overlap=OVERLAP)
        stitched, estimate = stitcher.stitch(frame)
        source = left_view[:, -OVERLAP:]
        left_edge, right_edge = _corrected_overlap(stitcher, frame, estimate)
        # The outer 4 columns are the fisheye rim, where resampling hits the
        # black pixels outside the disc. Score the seam inside that rim.
        left_agreement = _overlap_agreement(left_edge[:, :-4], source[:, :-4])
        right_agreement = _overlap_agreement(right_edge[:, 4:], source[:, 4:])
        self.assertGreaterEqual(left_agreement, 0.99, f"left seam vs source {left_agreement:.3f}")
        self.assertGreaterEqual(right_agreement, 0.99, f"right seam vs source {right_agreement:.3f}")
        self.assertEqual(stitched.shape[0], 180)
        self.assertEqual(stitched.shape[1], 360)
        self.assertAlmostEqual(estimate.shift_x, 0.0, delta=1.0)

    def test_recovers_shift_and_exposure_to_99_percent(self):
        stitcher = RealtimeStitcher(FOV, BASE, output_width=360, smoothing=0.0)
        _clean, source_view = _lenses(PROJ_H, PROJ_W, overlap=OVERLAP)
        shifted, _view = _lenses(PROJ_H, PROJ_W, overlap=OVERLAP, shift_x=6.0, right_scale=0.62)
        source = source_view[:, -OVERLAP:]

        _fixed, estimate = stitcher.stitch(shifted)
        fixed_l, fixed_r = _corrected_overlap(stitcher, shifted, estimate)

        self.assertAlmostEqual(abs(estimate.shift_x), 6.0, delta=1.0)
        self.assertAlmostEqual(estimate.gain, 1.0 / 0.62, delta=0.12)
        left_agreement = _overlap_agreement(fixed_l[:, :-4], source[:, :-4])
        right_agreement = _overlap_agreement(fixed_r[:, 4:], source[:, 4:])
        self.assertGreaterEqual(left_agreement, 0.99, f"left seam vs source {left_agreement:.3f}")
        self.assertGreaterEqual(
            right_agreement, 0.99, f"corrected right seam vs source {right_agreement:.3f}, shift={estimate}"
        )

    def test_edge_lighting_step_is_small(self):
        stitcher = RealtimeStitcher(FOV, BASE, output_width=360, smoothing=0.0)
        frame, _view = _lenses(PROJ_H, PROJ_W, overlap=OVERLAP, right_scale=0.55)
        # Extra vertical falloff on the right disc, strongest at the top.
        right = frame[:, frame.shape[1] // 2 :].astype(np.float32)
        ramp = np.linspace(0.75, 1.0, right.shape[0], dtype=np.float32)[:, None, None]
        frame = frame.copy()
        frame[:, frame.shape[1] // 2 :] = np.clip(right * ramp, 0, 255).astype(np.uint8)

        _stitched, estimate = stitcher.stitch(frame)
        left_edge, right_edge = _corrected_overlap(stitcher, frame, estimate)
        gray_l = left_edge.astype(np.float32).mean(axis=2)
        gray_r = right_edge.astype(np.float32).mean(axis=2)
        y0, y1 = int(gray_l.shape[0] * 0.25), int(gray_l.shape[0] * 0.75)
        step = np.abs(gray_l[y0:y1] - gray_r[y0:y1]).mean()
        self.assertLess(step, 8.0, f"lighting step across the seam is {step:.2f}")

    def test_frame_time_is_realtime(self):
        stitcher = RealtimeStitcher(FOV, BASE, output_width=360, smoothing=0.4)
        frame, _view = _lenses(PROJ_H, PROJ_W, overlap=OVERLAP, shift_x=3.0, right_scale=0.8)
        stitcher.stitch(frame)  # build maps
        start = time.perf_counter()
        for _ in range(5):
            stitcher.stitch(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5.0
        self.assertLess(elapsed_ms, 80.0, f"stitch took {elapsed_ms:.1f} ms")


class RenderVideoTests(unittest.TestCase):
    def test_each_output_format_writes_a_video(self):
        import tempfile

        from camera_calibration.realtime.stitcher import main, render_video

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 80)
            )
            self.assertTrue(writer.isOpened())
            for index in range(3):
                frame = np.full((80, 160, 3), 40 + index * 20, np.uint8)
                writer.write(frame)
            writer.release()

            fisheye = render_video(source)
            self.assertTrue(fisheye.endswith("clip_fisheye.mp4"))
            calib = Path(folder) / "calibration.toml"
            calib.write_text(
                "\n".join([
                    "[metadata]",
                    "fov_deg = 190",
                    "[lens1]",
                    "center_x = 0.5",
                    "center_y = 0.5",
                    "p1 = 0.0",
                    "[lens2]",
                    "center_x = 0.5",
                    "center_y = 0.5",
                    "p1 = 0.0",
                    "",
                ]),
                encoding="utf-8",
            )
            flat = render_video(source, output_format="rectalinear", calibration=calib)
            panorama = main([str(source), "--output_format", "360", "--calibration", str(calib)])

            for path, size in (
                (fisheye, (160, 80)),
                (flat, (160, 80)),
                (panorama, (160, 80)),
            ):
                capture = cv2.VideoCapture(path)
                self.assertTrue(capture.isOpened(), path)
                ok, frame = capture.read()
                capture.release()
                self.assertTrue(ok, path)
                self.assertEqual((frame.shape[1], frame.shape[0]), size)
            self.assertIn(b"equirectangular", Path(panorama).read_bytes())


if __name__ == "__main__":
    unittest.main()
