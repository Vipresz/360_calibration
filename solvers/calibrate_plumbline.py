#!/usr/bin/env python3
"""Plumb-line calibration for dual-fisheye 360 cameras.

Algorithm
---------
A straight line in the world projects to camera rays that are all coplanar:
they lie on a single great circle of the unit sphere.  For each candidate
lens model we:

  1. Detect line SEGMENTS in each fisheye half independently (LSD,
     FastLineDetector, or Canny + HoughLinesP).
  2. Sample N points along each segment, in fisheye pixel space.
  3. Project every sample to a 3-D world ray under the candidate lens model.
  4. For each segment, find the best-fit plane through the origin (smallest
     eigenvector of the rays' outer-product sum).  The residual is the
     signed distance of each ray to that plane (= ray dot plane_normal).
  5. ``scipy.optimize.least_squares`` minimises all residuals across
     segments / frames / lenses jointly, plus Tikhonov pulls.

Observability
-------------
Plumb-line residuals are invariant under any per-lens rotation: rotating
all rays of a segment rotates the fitted plane along with them, leaving
distances unchanged.  Therefore lens rotation (yaw / pitch / roll) is
**unobservable** from plumb-line data and is held fixed at the initial
value.  Only the per-lens INTRINSICS (center, fov, k1/k2/k3) are observable.

The two lenses are independent: lens-1 segments contribute residuals that
depend only on lens-1 parameters and vice-versa.  No seam region needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

try:
    from camera_calibration.calib.calibration_config import (
        CameraCalibration, LensCalibration,
    )
    from camera_calibration.solvers.calibrate_features import (
        fisheye_pixels_to_world_rays,
        _lens_to_vec, _vec_to_lens, _TOML_PARAM_NAMES,
        _config_to_init_and_bounds, _reg_weights_from_config,
        _make_preview,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import (
        CameraCalibration, LensCalibration,
    )
    from camera_calibration.solvers.calibrate_features import (
        fisheye_pixels_to_world_rays,
        _lens_to_vec, _vec_to_lens, _TOML_PARAM_NAMES,
        _config_to_init_and_bounds, _reg_weights_from_config,
        _make_preview,
    )


# ---------------------------------------------------------------------------
# Line-segment detectors
# ---------------------------------------------------------------------------

def _try_lsd_detector():
    """Built-in OpenCV LSD detector, available on 4.5.5+ (re-added after the
    license issue), some 4.x builds still ship it.  Returns the detector or
    None if not available."""
    try:
        return cv2.createLineSegmentDetector()
    except (AttributeError, cv2.error):
        return None


def _try_fld_detector():
    """opencv-contrib FastLineDetector.  Returns None if contrib not installed."""
    try:
        return cv2.ximgproc.createFastLineDetector(length_threshold=20)
    except (AttributeError, cv2.error):
        return None


def _detect_segments_lsd(gray: np.ndarray, lsd) -> np.ndarray:
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _detect_segments_fld(gray: np.ndarray, fld) -> np.ndarray:
    lines = fld.detect(gray)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _detect_segments_hough(gray: np.ndarray,
                            canny_low: int, canny_high: int,
                            min_length_px: int,
                            threshold: int,
                            max_gap_px: int) -> np.ndarray:
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0,
                             threshold=threshold,
                             minLineLength=min_length_px,
                             maxLineGap=max_gap_px)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _detect_segments(img: np.ndarray, *, detector: str,
                      min_length_px: int,
                      hough_canny_low: int, hough_canny_high: int,
                      hough_threshold: int, hough_max_gap_px: int,
                      _detector_cache: dict) -> Tuple[np.ndarray, str]:
    """Return (segments_Nx4, detector_used).

    ``detector`` may be 'lsd', 'fld', 'hough', or 'auto'.  We cache detector
    instances across calls so SIFT-style heavy initialisation only happens
    once per process.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    candidates = ['lsd', 'fld', 'hough'] if detector == 'auto' else [detector]
    last_err = None
    for cand in candidates:
        try:
            if cand == 'lsd':
                if 'lsd' not in _detector_cache:
                    _detector_cache['lsd'] = _try_lsd_detector()
                if _detector_cache['lsd'] is None:
                    last_err = "LSD not available in this OpenCV build"
                    continue
                segs = _detect_segments_lsd(gray, _detector_cache['lsd'])
                return segs, 'lsd'
            if cand == 'fld':
                if 'fld' not in _detector_cache:
                    _detector_cache['fld'] = _try_fld_detector()
                if _detector_cache['fld'] is None:
                    last_err = "FastLineDetector requires opencv-contrib"
                    continue
                segs = _detect_segments_fld(gray, _detector_cache['fld'])
                return segs, 'fld'
            if cand == 'hough':
                segs = _detect_segments_hough(
                    gray, hough_canny_low, hough_canny_high,
                    min_length_px, hough_threshold, hough_max_gap_px)
                return segs, 'hough'
        except cv2.error as exc:
            last_err = str(exc)

    raise RuntimeError(
        f"No line-segment detector worked.  Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Segment filtering & sampling
# ---------------------------------------------------------------------------

def _segment_lengths(segments: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.empty(0, dtype=np.float64)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    return np.sqrt(dx * dx + dy * dy)


def _filter_segments(segments: np.ndarray,
                      *,
                      min_length_px: int,
                      max_segments: int,
                      lens_cx: float, lens_cy: float, lens_radius: float,
                      edge_margin_ratio: float) -> np.ndarray:
    """Drop segments that are too short or that touch the fisheye disk edge.

    Returns the longest ``max_segments`` survivors, sorted by length descending
    (so the most informative segments come first).
    """
    if len(segments) == 0:
        return segments

    lengths = _segment_lengths(segments)
    keep = lengths >= float(min_length_px)
    segments = segments[keep]
    lengths = lengths[keep]
    if len(segments) == 0:
        return segments

    # Drop segments whose endpoints lie outside (or right on the rim of) the
    # fisheye disk -- those pixels back-project past the FOV horizon and the
    # ray model gets unreliable.
    inner_r = lens_radius * (1.0 - max(0.0, edge_margin_ratio))
    d1 = np.sqrt((segments[:, 0] - lens_cx) ** 2 + (segments[:, 1] - lens_cy) ** 2)
    d2 = np.sqrt((segments[:, 2] - lens_cx) ** 2 + (segments[:, 3] - lens_cy) ** 2)
    keep = (d1 < inner_r) & (d2 < inner_r)
    segments = segments[keep]
    lengths = lengths[keep]
    if len(segments) == 0:
        return segments

    if len(segments) > max_segments:
        order = np.argsort(-lengths)[:max_segments]
        segments = segments[order]
    return segments


def _sample_segment_pixels(segments: np.ndarray,
                            n_samples: int) -> np.ndarray:
    """For each (x1, y1, x2, y2) in ``segments``, return ``n_samples`` evenly-
    spaced points along the segment.  Output shape: (N_segs, n_samples, 2)."""
    if len(segments) == 0 or n_samples <= 0:
        return np.empty((0, n_samples, 2), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    starts = segments[:, :2].astype(np.float64)
    ends   = segments[:, 2:].astype(np.float64)
    deltas = ends - starts
    pts = starts[:, None, :] + t[None, :, None] * deltas[:, None, :]
    return pts


# ---------------------------------------------------------------------------
# Plane-fit residuals
# ---------------------------------------------------------------------------

def _segment_plane_residuals(rays: np.ndarray) -> np.ndarray:
    """Per-segment plane fit + signed distances.

    ``rays`` has shape (N_segs, n_samples, 3) of unit world rays.
    Returns shape (N_segs, n_samples) of signed distances (= rays . normal).

    For each segment we compute  M = sum_i  r_i r_i^T   (3x3),
    take the eigenvector of M's smallest eigenvalue as the plane normal, and
    project every ray onto that normal.

    A perfect straight world line yields a flat residual; the optimiser
    drives lens parameters to flatten this for every segment simultaneously.
    """
    if rays.size == 0:
        return np.empty((0,), dtype=np.float64)

    # Outer products: M[i] = rays[i].T @ rays[i] in batch.
    # rays[i].T @ rays[i] = einsum('sj,sk->jk', rays[i], rays[i])
    M = np.einsum('isj,isk->ijk', rays, rays)  # (N_segs, 3, 3)

    # Symmetric -> eigh.  Smallest eigenvalue is the first column.
    _eigvals, eigvecs = np.linalg.eigh(M)
    normals = eigvecs[..., 0]  # (N_segs, 3)

    # Signed distance per sample = ray dot normal.
    return np.einsum('isj,ij->is', rays, normals)


# ---------------------------------------------------------------------------
# Pre-compute pixels once (rays change every iteration; pixels don't)
# ---------------------------------------------------------------------------

class _PerLensSamples:
    """Pre-computed per-lens sample data.

    Holds a (N_total_samples, 2) array of fisheye pixel coords plus a
    (N_segs+1,) array of segment-boundary indices, so that an optimiser
    iteration just back-projects the flat pixel array, reshapes to
    (N_segs, n_samples, 3), and runs the plane fit.
    """

    def __init__(self, pixel_groups: List[np.ndarray]):
        # pixel_groups: list of (n_samples, 2) arrays, one per surviving segment.
        if pixel_groups:
            self.n_segs = len(pixel_groups)
            self.n_samples = pixel_groups[0].shape[0]
            self.pixels_flat = np.concatenate(pixel_groups, axis=0)
        else:
            self.n_segs = 0
            self.n_samples = 0
            self.pixels_flat = np.empty((0, 2), dtype=np.float64)

    @property
    def n_residuals(self) -> int:
        return self.n_segs * self.n_samples

    def rays_for_lens(self, lens: LensCalibration, w: int, h: int,
                       *, flip: bool) -> np.ndarray:
        """Return shape (N_segs, n_samples, 3) of unit world rays."""
        if self.n_segs == 0:
            return np.empty((0, self.n_samples, 3), dtype=np.float64)
        flat_rays = fisheye_pixels_to_world_rays(self.pixels_flat, lens, w, h,
                                                   flip=flip)
        return flat_rays.reshape(self.n_segs, self.n_samples, 3)


# ---------------------------------------------------------------------------
# Detection orchestration
# ---------------------------------------------------------------------------

def _collect_samples_for_lens(images: Sequence[np.ndarray],
                                lens: LensCalibration,
                                *,
                                detector: str,
                                min_length_px: int,
                                max_segments_per_half: int,
                                samples_per_segment: int,
                                edge_margin_ratio: float,
                                hough_canny_low: int,
                                hough_canny_high: int,
                                hough_threshold: int,
                                hough_max_gap_px: int,
                                detector_cache: dict,
                                lens_label: str = '',
                                verbose: bool = True) -> _PerLensSamples:
    """Detect + filter segments across all frames for a single lens half."""
    if not images:
        return _PerLensSamples([])

    h0, w0 = images[0].shape[:2]
    cx_px = lens.center_x * w0
    cy_px = lens.center_y * h0
    radius = min(w0, h0) / 2.0

    pixel_groups: List[np.ndarray] = []
    detectors_used: dict = {}
    raw_total = 0

    for f_idx, img in enumerate(images):
        segments, used = _detect_segments(
            img,
            detector=detector,
            min_length_px=min_length_px,
            hough_canny_low=hough_canny_low,
            hough_canny_high=hough_canny_high,
            hough_threshold=hough_threshold,
            hough_max_gap_px=hough_max_gap_px,
            _detector_cache=detector_cache,
        )
        raw_total += len(segments)
        detectors_used[used] = detectors_used.get(used, 0) + 1

        segments = _filter_segments(
            segments,
            min_length_px=min_length_px,
            max_segments=max_segments_per_half,
            lens_cx=cx_px, lens_cy=cy_px, lens_radius=radius,
            edge_margin_ratio=edge_margin_ratio,
        )
        if len(segments) == 0:
            continue
        sampled = _sample_segment_pixels(segments, samples_per_segment)
        for s in sampled:
            pixel_groups.append(s)

    if verbose:
        det_summary = ", ".join(f"{name}x{n}" for name, n in detectors_used.items())
        print(f"  {lens_label}: {len(pixel_groups):4d} segments kept "
              f"(raw={raw_total}, samples/seg={samples_per_segment}, "
              f"detector={det_summary or '-'})")

    return _PerLensSamples(pixel_groups)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calibrate_plumbline(frames_data: Sequence[Tuple[np.ndarray, np.ndarray]],
                         base_fov: float,
                         *,
                         init_lens1: Optional[LensCalibration] = None,
                         init_lens2: Optional[LensCalibration] = None,
                         config=None,
                         max_iter: int = 200,
                         preview: bool = False,
                         preview_window: str = 'plumbline calibration preview',
                         verbose: bool = True) -> CameraCalibration:
    """Run plumb-line calibration.

    Args:
        frames_data: iterable of (left_img, right_img) BGR pairs.
        base_fov:    only used for the live preview render; does not affect
                     the residual itself (residuals live in 3-D ray space).
        init_lens1, init_lens2: starting lens guess.  Ignored when ``config``
                     supplies them.  Defaults to a centred 185 deg lens.
        config:      ``calibrate_adjoint.OptimizationConfig`` parsed from
                     plumbline_tuning.toml.  Provides:
                       * lens1/lens2 initial values (`nominal`) and bounds;
                       * `optimize=false` parameters are pinned;
                       * lens2 entries with `use_lens1=true` mirror lens1
                         after every residual evaluation;
                       * a [plumbline] section with detector settings;
                       * a [regularization] section with Tikhonov weights.
        max_iter:    base iteration budget; final ``max_nfev = min(max_iter*20, 600)``.
        preview:     show the stitched first frame in an OpenCV window,
                     refreshed every accepted optimisation step.
    """
    # ---- Pull initial values + bounds + ties from config -------------------
    if config is not None:
        init_lens1, init_lens2, lo_cfg, hi_cfg, tied_pairs = \
            _config_to_init_and_bounds(config)
    else:
        if init_lens1 is None:
            init_lens1 = LensCalibration(0.5, 0.5, 185.0)
        if init_lens2 is None:
            init_lens2 = LensCalibration(0.5, 0.5, 185.0)
        # Built-in fallback bounds: tight on rotations (unobservable here),
        # moderate on intrinsics.
        EPS = 1e-9
        lo_list, hi_list = [], []
        for L in (init_lens1, init_lens2):
            lo_list += [L.center_x - 0.04, L.center_y - 0.04, max(170.0, L.fov - 15.0),
                        -0.20, -0.20, -0.10,
                        L.rotation_yaw - EPS, L.rotation_pitch - EPS, L.rotation_roll - EPS]
            hi_list += [L.center_x + 0.04, L.center_y + 0.04, min(210.0, L.fov + 15.0),
                         0.20,  0.20,  0.10,
                        L.rotation_yaw + EPS, L.rotation_pitch + EPS, L.rotation_roll + EPS]
        lo_cfg = np.asarray(lo_list, dtype=np.float64)
        hi_cfg = np.asarray(hi_list, dtype=np.float64)
        tied_pairs = []

    # Plumb-line residuals are invariant under rotation -- if the user
    # accidentally enabled rotations in the config, warn (they'll drift
    # toward whatever balances the regulariser, contributing nothing useful).
    if config is not None:
        for lens in ('lens1', 'lens2'):
            for rname in ('rotation_yaw', 'rotation_pitch', 'rotation_roll'):
                if config.params[f"{lens}.{rname}"]['optimize']:
                    print(f"  WARNING: {lens}.{rname} optimize=true, but plumb-line "
                          f"residuals are invariant under rotation -- this DOF will "
                          f"drift along the regulariser.  Set optimize=false.")

    # ---- Pull plumb-line section knobs -------------------------------------
    pl_cfg = getattr(config, 'plumbline', None) if config is not None else None
    pl_cfg = pl_cfg or {}
    detector              = str(pl_cfg.get('detector', 'auto'))
    min_length_px         = int(pl_cfg.get('min_length_px',         40))
    max_segments_per_half = int(pl_cfg.get('max_segments_per_half', 200))
    samples_per_segment   = int(pl_cfg.get('samples_per_segment',     8))
    edge_margin_ratio     = float(pl_cfg.get('edge_margin_ratio',  0.05))
    hough_canny_low       = int(pl_cfg.get('hough_canny_low',        50))
    hough_canny_high      = int(pl_cfg.get('hough_canny_high',      150))
    hough_threshold       = int(pl_cfg.get('hough_threshold',        60))
    hough_max_gap_px      = int(pl_cfg.get('hough_max_gap_px',        8))
    cfg_xtol              = float(pl_cfg.get('xtol',     1e-6))
    cfg_ftol              = float(pl_cfg.get('ftol',     1e-7))
    cfg_gtol              = float(pl_cfg.get('gtol',     1e-8))
    cfg_max_nfev          = int(pl_cfg.get('max_nfev',    600))

    reg_weights = _reg_weights_from_config(config)

    # ---- Detect segments per lens -----------------------------------------
    if not frames_data:
        raise ValueError("calibrate_plumbline: no frames provided")
    h, w = frames_data[0][0].shape[:2]

    lefts  = [f[0] for f in frames_data]
    rights = [f[1] for f in frames_data]

    if verbose:
        print()
        print("=" * 60)
        print(f"PLUMB-LINE CALIBRATION  (detector={detector})")
        print("=" * 60)
        print(f"  Frames        : {len(frames_data)}")
        print(f"  Frame size    : {w}x{h} per half")
        print(f"  Min seg length: {min_length_px} px")
        print(f"  Samples/seg   : {samples_per_segment}")
        print(f"  Max segs/half : {max_segments_per_half}")
        print(f"  Edge margin   : {edge_margin_ratio:.2f} of disk radius")

    detector_cache: dict = {}
    samples1 = _collect_samples_for_lens(
        lefts, init_lens1, detector=detector,
        min_length_px=min_length_px,
        max_segments_per_half=max_segments_per_half,
        samples_per_segment=samples_per_segment,
        edge_margin_ratio=edge_margin_ratio,
        hough_canny_low=hough_canny_low, hough_canny_high=hough_canny_high,
        hough_threshold=hough_threshold, hough_max_gap_px=hough_max_gap_px,
        detector_cache=detector_cache,
        lens_label='lens 1',
        verbose=verbose,
    )
    # Lens 2 input is mirrored before projection (matches the fliplr
    # convention used in stitch_dual_fisheye); detect on the mirrored
    # version so the resulting pixel coords match what
    # fisheye_pixels_to_world_rays expects with flip=True.
    rights_flipped = [np.fliplr(img) for img in rights]
    samples2 = _collect_samples_for_lens(
        rights_flipped, init_lens2, detector=detector,
        min_length_px=min_length_px,
        max_segments_per_half=max_segments_per_half,
        samples_per_segment=samples_per_segment,
        edge_margin_ratio=edge_margin_ratio,
        hough_canny_low=hough_canny_low, hough_canny_high=hough_canny_high,
        hough_threshold=hough_threshold, hough_max_gap_px=hough_max_gap_px,
        detector_cache=detector_cache,
        lens_label='lens 2',
        verbose=verbose,
    )

    n1 = samples1.n_residuals
    n2 = samples2.n_residuals
    n_data = n1 + n2

    if samples1.n_segs == 0 and samples2.n_segs == 0:
        raise RuntimeError(
            "Plumb-line solver: no usable segments detected in any lens.\n"
            "  - Lower [plumbline].min_length_px (try 20-30) or relax the\n"
            "    hough_threshold; or use --detector lsd if you have OpenCV >= 4.5.5.\n"
            "  - Make sure the scene has actual straight edges (architecture,\n"
            "    doors, panels, books).  Empty-room footage is hopeless."
        )
    if samples1.n_segs < 5 or samples2.n_segs < 5:
        print(f"  WARNING: lens 1 has {samples1.n_segs} segments, "
              f"lens 2 has {samples2.n_segs} -- one lens will be poorly "
              f"constrained.  Consider feeding more frames or a more "
              f"line-rich scene.")

    if verbose:
        print(f"  Total residuals: lens1={n1}, lens2={n2}, total={n_data}")

    # ---- Build initial vector + clip to bounds -----------------------------
    x0 = np.concatenate([_lens_to_vec(init_lens1), _lens_to_vec(init_lens2)])
    x0_clipped = np.clip(x0, lo_cfg, hi_cfg)

    # Apply ties to x0 so lens2 mirror is consistent from step 0.
    for i2, i1 in tied_pairs:
        x0_clipped[i2] = x0_clipped[i1]

    # ---- Live preview -----------------------------------------------------
    preview_first_pair = frames_data[0] if (preview and frames_data) else None
    preview_window_open = False
    if preview_first_pair is not None:
        try:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            cv2.imshow(
                preview_window,
                _make_preview(*preview_first_pair, init_lens1, init_lens2, base_fov),
            )
            cv2.waitKey(1)
            preview_window_open = True
        except cv2.error:
            preview_window_open = False

    # ---- Residual fn ------------------------------------------------------
    iter_count = [0]
    best_rms = [float('inf')]
    x0_for_reg = x0_clipped.copy()

    def _residuals(x):
        # Apply ties (use_lens1) before evaluating.
        if tied_pairs:
            x = x.copy()
            for i2, i1 in tied_pairs:
                x[i2] = x[i1]
        lens1 = _vec_to_lens(x[:9])
        lens2 = _vec_to_lens(x[9:])
        rays1 = samples1.rays_for_lens(lens1, w, h, flip=False)
        rays2 = samples2.rays_for_lens(lens2, w, h, flip=True)
        # The plane fit naturally normalises by the segment count, so the
        # RMS over (rays.dot(normal)) is the per-sample distance to the
        # best-fit plane in unit-ray space.
        r1 = _segment_plane_residuals(rays1).reshape(-1) if samples1.n_segs > 0 \
                else np.empty(0, dtype=np.float64)
        r2 = _segment_plane_residuals(rays2).reshape(-1) if samples2.n_segs > 0 \
                else np.empty(0, dtype=np.float64)
        data_res = np.concatenate([r1, r2])

        weights = np.tile(reg_weights, 2)
        reg_res = (x - x0_for_reg) * weights
        r = np.concatenate([data_res, reg_res])

        iter_count[0] += 1
        rms = float(np.sqrt(np.mean(data_res ** 2))) if data_res.size > 0 else 0.0
        if rms < best_rms[0] * 0.999:
            best_rms[0] = rms
            ang = float(np.degrees(np.arcsin(np.clip(rms, 0.0, 1.0))))
            if verbose:
                print(f"    step (eval {iter_count[0]:4d}): plane RMS={rms:.5f}  "
                      f"angular={ang:.4f} deg")
            if preview_window_open and preview_first_pair is not None:
                try:
                    cv2.imshow(
                        preview_window,
                        _make_preview(*preview_first_pair, lens1, lens2, base_fov))
                    cv2.waitKey(1)
                except cv2.error:
                    pass
        return r

    if verbose:
        print(f"\n  Optimising 18 parameters over {n_data} residuals "
              f"(+ 18 regulariser rows) ...")

    res = least_squares(
        _residuals, x0_clipped,
        bounds=(lo_cfg, hi_cfg),
        method='trf',
        x_scale='jac',
        max_nfev=min(max_iter * 20, cfg_max_nfev),
        xtol=cfg_xtol, ftol=cfg_ftol, gtol=cfg_gtol,
        verbose=0,
    )

    final_data = res.fun[:n_data]
    final_rms = float(np.sqrt(np.mean(final_data ** 2))) if final_data.size > 0 else 0.0
    final_ang = float(np.degrees(np.arcsin(np.clip(final_rms, 0.0, 1.0))))
    if verbose:
        print(f"  Final  plane RMS: {final_rms:.5f}  "
              f"(angular {final_ang:.3f} deg)")
        print(f"  scipy: {res.message}, {res.nfev} evals, status {res.status}")

    x_final = res.x.copy()
    for i2, i1 in tied_pairs:
        x_final[i2] = x_final[i1]
    out_lens1 = _vec_to_lens(x_final[:9])
    out_lens2 = _vec_to_lens(x_final[9:])

    if verbose:
        print()
        print("=" * 60)
        print("OPTIMIZED PARAMETERS")
        print("=" * 60)
        for label, L in (('Lens 1', out_lens1), ('Lens 2', out_lens2)):
            print(f"\n{label}:")
            print(f"  center: ({L.center_x:.6f}, {L.center_y:.6f})")
            print(f"  fov:    {L.fov:.2f} deg")
            print(f"  k:      [{L.k1:+.6f}, {L.k2:+.6f}, {L.k3:+.6f}]")
            print(f"  rot:    [{L.rotation_yaw:+.6f}, {L.rotation_pitch:+.6f}, "
                  f"{L.rotation_roll:+.6f}]  (unobservable; kept at init)")

    if preview_window_open:
        try:
            cv2.destroyWindow(preview_window)
        except cv2.error:
            pass

    return CameraCalibration(lens1=out_lens1, lens2=out_lens2, is_horizontal=True)
