#!/usr/bin/env python3
"""Joint optimization calibration for dual-fisheye 360° cameras."""

import argparse
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import differential_evolution

try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        tomllib = None

try:
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, stitch_dual_fisheye
    )
    from camera_calibration.transformations.edges import extract_edges
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, stitch_dual_fisheye
    )
    from camera_calibration.transformations.edges import extract_edges

# Default tuning file lives next to the calib/ package
# (configs/adjoint_tuning.toml relative to this file in solvers/).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "adjoint_tuning.toml"

# Sentinel "missing" — distinguishes "not set in config" from "set to None"
_MISSING = object()


class OptimizationConfig:
    """Manages optimization parameters from TOML config."""
    
    PARAM_NAMES = [
        'lens1.center_x', 'lens1.center_y', 'lens1.fov',
        'lens1.k1', 'lens1.k2', 'lens1.k3',
        'lens1.rotation_yaw', 'lens1.rotation_pitch', 'lens1.rotation_roll',
        'lens2.center_x', 'lens2.center_y', 'lens2.fov',
        'lens2.k1', 'lens2.k2', 'lens2.k3',
        'lens2.rotation_yaw', 'lens2.rotation_pitch', 'lens2.rotation_roll',
    ]
    
    # ---- Defaults (kept in sync with solvers/configs/adjoint_tuning.toml) -----
    # Rotations are in RADIANS (~0.15 rad ≈ 8.6°).
    _DEFAULT_PARAMS = {
        'lens1.center_x':       {'optimize': True,  'nominal': 0.5,   'min':  0.48, 'max':  0.52},
        'lens1.center_y':       {'optimize': True,  'nominal': 0.5,   'min':  0.48, 'max':  0.52},
        'lens1.fov':            {'optimize': True,  'nominal': 195.0, 'min': 185.0, 'max': 200.0},
        'lens1.k1':             {'optimize': True,  'nominal': 0.0,   'min': -0.1,  'max':  0.1},
        'lens1.k2':             {'optimize': True,  'nominal': 0.0,   'min': -0.1,  'max':  0.1},
        'lens1.k3':             {'optimize': False, 'nominal': 0.0,   'min': -0.1,  'max':  0.1},
        'lens1.rotation_yaw':   {'optimize': False, 'nominal': 0.0,   'min': -0.15, 'max':  0.15},
        'lens1.rotation_pitch': {'optimize': False, 'nominal': 0.0,   'min': -0.15, 'max':  0.15},
        'lens1.rotation_roll':  {'optimize': False, 'nominal': 0.0,   'min': -0.15, 'max':  0.15},
        'lens2.center_x':       {'optimize': True,  'nominal': 0.5,   'min':  0.48, 'max':  0.52},
        'lens2.center_y':       {'optimize': True,  'nominal': 0.5,   'min':  0.48, 'max':  0.52},
        'lens2.fov':            {'optimize': False, 'nominal': 195.0, 'min': 185.0, 'max': 200.0, 'use_lens1': True},
        'lens2.k1':             {'optimize': True,  'nominal': 0.0,   'min': -0.3,  'max':  0.3},
        'lens2.k2':             {'optimize': True,  'nominal': 0.0,   'min': -0.3,  'max':  0.3},
        'lens2.k3':             {'optimize': False, 'nominal': 0.0,   'min': -0.1,  'max':  0.1},
        'lens2.rotation_yaw':   {'optimize': True,  'nominal': 0.0,   'min': -0.15, 'max':  0.15},
        'lens2.rotation_pitch': {'optimize': True,  'nominal': 0.0,   'min': -0.10, 'max':  0.10},
        'lens2.rotation_roll':  {'optimize': False, 'nominal': 0.0,   'min': -0.10, 'max':  0.10},
    }

    _DEFAULT_OPTIMIZER = {
        'strategy': 'best1bin', 'maxiter': 100, 'popsize': 15,
        'tol': 0.01, 'mutation': [0.5, 1.0], 'recombination': 0.7,
    }
    _DEFAULT_PREPROCESSING = {
        'normalize_lighting':  True,
        'edge_margin_ratio':   0.05,
        'vignette_correction': True,    # equalize radial brightness fall-off
        'vignette_strength':   1.0,     # 0.0 = off, 1.0 = fully flatten
        'comparison_mode':     'intensity',  # "intensity" | "edges"
        'edge_method':         'canny',      # "sobel" | "scharr" | "laplacian" | "canny"
        # Seam comparison strip in FOV space (see compute_seam_error).
        # ``seam_overlap_deg``: total width of the strip in degrees of FOV.
        #     0 = use full geometric overlap (legacy: FOV [base_fov, lens_fov]).
        # ``seam_overlap_center_deg``: centre of the strip in degrees of FOV.
        #     0 = use ``base_fov`` (the panorama hemisphere boundary).  Set this
        #     to a fixed value (e.g. 185) to anchor the strip away from
        #     base_fov.  The centre is FIXED -- it does NOT track the dynamic
        #     ``lens_fov`` -- otherwise the optimiser can game the metric by
        #     shifting lens_fov and dragging the strip onto a different patch.
        'seam_overlap_deg':         0.0,
        'seam_overlap_center_deg':  0.0,
    }
    _DEFAULT_REGULARIZATION = {
        'enabled': True,
        'lambda_center': 100.0, 'lambda_fov': 0.1,
        'lambda_distortion': 10.0, 'lambda_rotation': 50.0,
    }

    # ``[plumbline]`` is a plumbline-solver-only section.  Same idea as the
    # [features] block below: parsed unconditionally so a single
    # OptimizationConfig can serve every solver.
    _DEFAULT_PLUMBLINE = {
        'detector':              'auto',  # 'lsd' | 'fld' | 'hough' | 'auto'
        'min_length_px':         40,
        'max_segments_per_half': 200,
        'samples_per_segment':   8,
        'edge_margin_ratio':     0.05,
        'hough_canny_low':       50,
        'hough_canny_high':      150,
        'hough_threshold':       60,
        'hough_max_gap_px':      8,
        # scipy.least_squares tolerances and budget.  Defaults below are tight-
        # ish; loosen tols (1e-9) or raise max_nfev to make the solver iterate
        # longer when starting near a local minimum.
        'xtol':           1e-6,
        'ftol':           1e-7,
        'gtol':           1e-8,
        'max_nfev':       600,
    }

    # ``[features]`` is a features-solver-only section.  We parse it here so a
    # single OptimizationConfig instance can serve both solvers; the adjoint
    # solver simply ignores it.
    _DEFAULT_FEATURES = {
        'detector':    'sift',
        'ratio':        0.85,
        'ransac_deg':   5.0,
        'min_matches':  12,
        # Optional override for where SIFT/ORB looks for keypoints.
        # Same semantics as adjoint_tuning.toml's seam_overlap_*:
        #   overlap_deg = 0.0  -> use the geometric dual-lens overlap
        #                          (lens.fov - 180 deg per side).
        #   overlap_deg > 0    -> strip is FOV [center - w/2, center + w/2].
        #   overlap_center_deg = 0  -> centre defaults to base_fov (180 deg,
        #                              i.e. the hemisphere boundary).
        'overlap_deg':         0.0,
        'overlap_center_deg':  0.0,
    }
    # Features-solver Tikhonov pull weights.  Per-parameter, in the same order
    # as the 9-vector layout used by calibrate_features._lens_to_vec.
    _DEFAULT_FEATURES_REG = {
        'enabled': True,
        'weight_center_x':       20.0,
        'weight_center_y':       20.0,
        'weight_fov':             0.05,
        'weight_k1':              3.0,
        'weight_k2':              3.0,
        'weight_k3':             10.0,
        'weight_rotation_yaw':    3.0,
        'weight_rotation_pitch':  3.0,
        'weight_rotation_roll':   3.0,
    }

    def __init__(self, config_path=None, *, require_config=False):
        """Build configuration from a TOML file (or fall back to defaults).

        Args:
            config_path: Path to a tuning TOML.  If None or empty, defaults are used.
            require_config: When True, an explicit ``config_path`` that does not
                exist (or fails to load) raises FileNotFoundError instead of
                silently falling back.  Use this when the user passed --config
                on the command line.
        """
        self.params = {}
        self.optimizer = {}
        self.preprocessing = {}
        self.regularization = {}
        # Features-solver-specific knobs (only populated when the TOML has a
        # [features] section; otherwise they hold sensible defaults so the
        # features solver can read them unconditionally).
        self.features = dict(self._DEFAULT_FEATURES)
        self.features_regularization = dict(self._DEFAULT_FEATURES_REG)
        # Plumb-line-solver-specific knobs (only populated by [plumbline]).
        self.plumbline = dict(self._DEFAULT_PLUMBLINE)
        self.warnings = []

        if config_path:
            p = Path(config_path)
            if not p.exists():
                msg = f"Config file not found: {config_path}"
                if require_config:
                    raise FileNotFoundError(msg)
                print(f"Warning: {msg} -- using built-in defaults")
                self._use_defaults()
            else:
                self._load_toml(p)
        else:
            self._use_defaults()

        self._validate()
        self._build_param_mapping()

    def _load_toml(self, config_path):
        if tomllib is None:
            if hasattr(self, '_require_config'):
                raise RuntimeError("No TOML parser available (need Python 3.11+ or `pip install toml`).")
            print("Warning: TOML not available, using defaults")
            self._use_defaults()
            return

        # tomllib (stdlib) wants binary; the legacy `toml` package wants text.
        # Distinguish the two by TypeError, NOT bare Exception, so a real
        # TOMLDecodeError surfaces with its line number.
        try:
            with open(config_path, 'rb') as f:
                config = tomllib.load(f)
        except TypeError:
            with open(config_path, 'r', encoding='utf-8') as ft:
                config = tomllib.load(ft)

        for lens in ['lens1', 'lens2']:
            lens_config = config.get(lens, {})
            for param in ['center_x', 'center_y', 'fov', 'k1', 'k2', 'k3',
                          'rotation_yaw', 'rotation_pitch', 'rotation_roll']:
                key = f"{lens}.{param}"
                p = lens_config.get(param, _MISSING)
                default = self._DEFAULT_PARAMS[key]
                if p is _MISSING:
                    # Section omitted entirely -- fall back to compiled defaults.
                    self.params[key] = dict(default)
                    continue
                entry = {
                    'optimize':  p.get('optimize',  default['optimize']),
                    'nominal':   p.get('nominal',   default['nominal']),
                    'min':       p.get('min',       default['min']),
                    'max':       p.get('max',       default['max']),
                    'use_lens1': p.get('use_lens1', default.get('use_lens1', False)),
                }
                # Warn if optimize=true but bounds are missing (would silently use [-1,1])
                if entry['optimize'] and ('min' not in p or 'max' not in p):
                    self.warnings.append(
                        f"{key}: optimize=true but min/max not set -- using defaults "
                        f"[{entry['min']}, {entry['max']}]"
                    )
                self.params[key] = entry

        # Merge optimizer / preprocessing / regularization with defaults
        self.optimizer = {**self._DEFAULT_OPTIMIZER, **config.get('optimizer', {})}
        self.preprocessing = {**self._DEFAULT_PREPROCESSING, **config.get('preprocessing', {})}
        # [regularization] is overloaded between the two solvers: the adjoint
        # uses lambda_*, the features solver uses weight_*.  Merge them onto
        # both default dicts and let each solver pluck what it understands.
        raw_reg = config.get('regularization', {})
        self.regularization = {**self._DEFAULT_REGULARIZATION, **raw_reg}
        self.features_regularization = {**self._DEFAULT_FEATURES_REG, **raw_reg}
        # [features] is opt-in; absent in adjoint_tuning.toml.
        self.features = {**self._DEFAULT_FEATURES, **config.get('features', {})}
        # [plumbline] is opt-in too.
        self.plumbline = {**self._DEFAULT_PLUMBLINE, **config.get('plumbline', {})}

        print(f"Loaded config: {config_path}")
        # One-line sanity check: TOML must spell the key exactly ``optimize`` (not
        # ``ptimize`` etc.); otherwise the loader falls back to the default flag.
        rot_opt = [
            f"{l}.{p}={'on' if self.params[f'{l}.{p}']['optimize'] else 'off'}"
            for l in ('lens1', 'lens2')
            for p in ('rotation_yaw', 'rotation_pitch', 'rotation_roll')
        ]
        print("  rotation optimize flags: " + ", ".join(rot_opt))

    def _use_defaults(self):
        for name, vals in self._DEFAULT_PARAMS.items():
            entry = dict(vals)
            entry.setdefault('use_lens1', False)
            self.params[name] = entry
        self.optimizer = dict(self._DEFAULT_OPTIMIZER)
        self.preprocessing = dict(self._DEFAULT_PREPROCESSING)
        self.regularization = dict(self._DEFAULT_REGULARIZATION)
        self.features = dict(self._DEFAULT_FEATURES)
        self.features_regularization = dict(self._DEFAULT_FEATURES_REG)
        self.plumbline = dict(self._DEFAULT_PLUMBLINE)

    def _validate(self):
        """Surface footguns: bad use_lens1 combinations, malformed bounds."""
        for name, p in self.params.items():
            if p.get('use_lens1'):
                if not name.startswith('lens2.'):
                    self.warnings.append(f"{name}: use_lens1=true is only meaningful for lens2.* params (ignored)")
                if p.get('optimize'):
                    self.warnings.append(
                        f"{name}: use_lens1=true together with optimize=true -- "
                        f"the use_lens1 link will be IGNORED while optimizing"
                    )
            if p['min'] > p['max']:
                self.warnings.append(f"{name}: min ({p['min']}) > max ({p['max']}) -- swapping")
                p['min'], p['max'] = p['max'], p['min']
            if not (p['min'] <= p['nominal'] <= p['max']):
                self.warnings.append(
                    f"{name}: nominal ({p['nominal']}) outside bounds [{p['min']}, {p['max']}]"
                )
        for w in self.warnings:
            print(f"Warning: {w}")
    
    def _build_param_mapping(self):
        self.opt_params = []
        self.opt_indices = {}
        for name in self.PARAM_NAMES:
            p = self.params[name]
            if p['optimize']:
                self.opt_indices[name] = len(self.opt_params)
                self.opt_params.append((name, (p['min'], p['max'])))
    
    def get_bounds(self):
        return [bounds for _, bounds in self.opt_params]
    
    def get_n_params(self):
        return len(self.opt_params)
    
    def get_initial_vector(self):
        return [self.params[name]['nominal'] for name, _ in self.opt_params]
    
    def vector_to_lenses(self, opt_vector):
        values = {name: self.params[name]['nominal'] for name in self.PARAM_NAMES}
        for i, (name, _) in enumerate(self.opt_params):
            values[name] = opt_vector[i]
        for name, p in self.params.items():
            if p.get('use_lens1') and not p['optimize']:
                values[name] = values[name.replace('lens2.', 'lens1.')]
        
        lens1 = LensCalibration(
            center_x=values['lens1.center_x'], center_y=values['lens1.center_y'],
            fov=values['lens1.fov'], k1=values['lens1.k1'], k2=values['lens1.k2'], k3=values['lens1.k3'],
            rotation_yaw=values['lens1.rotation_yaw'], rotation_pitch=values['lens1.rotation_pitch'],
            rotation_roll=values['lens1.rotation_roll'],
        )
        lens2 = LensCalibration(
            center_x=values['lens2.center_x'], center_y=values['lens2.center_y'],
            fov=values['lens2.fov'], k1=values['lens2.k1'], k2=values['lens2.k2'], k3=values['lens2.k3'],
            rotation_yaw=values['lens2.rotation_yaw'], rotation_pitch=values['lens2.rotation_pitch'],
            rotation_roll=values['lens2.rotation_roll'],
        )
        return lens1, lens2
    
    def compute_regularization(self, lens1, lens2):
        """Compute regularization: pulls each parameter toward its OWN nominal.

        E = lambda_c * sum ||c_i - c_i_nom||^2 + lambda_f * sum (fov_i - fov_i_nom)^2
          + lambda_k * sum k_i^2             + lambda_r * sum rot_i^2
        """
        reg = self.regularization
        breakdown = {'center': 0.0, 'fov': 0.0, 'distortion': 0.0, 'rotation': 0.0}

        if not reg.get('enabled', True):
            return 0.0, breakdown

        def _nom(name):
            return self.params[name]['nominal']

        breakdown['center'] = reg['lambda_center'] * (
            (lens1.center_x - _nom('lens1.center_x'))**2 +
            (lens1.center_y - _nom('lens1.center_y'))**2 +
            (lens2.center_x - _nom('lens2.center_x'))**2 +
            (lens2.center_y - _nom('lens2.center_y'))**2
        )
        # FOV: lens2 may be tied to lens1 (use_lens1).  In that case there is only
        # ONE physical FOV shared by both sensors -- regularizing (l1 - n1)^2
        # AND (l2 - n2)^2 double-counts the same DOF and, worse, n1 and n2 often
        # differ (e.g. 185 vs 195 in adjoint_tuning.toml), which explodes the penalty
        # even when l1 == l2 == a reasonable value.
        p2_fov = self.params['lens2.fov']
        if p2_fov.get('use_lens1') and not p2_fov['optimize']:
            fov_sq_sum = (lens1.fov - _nom('lens1.fov'))**2
        else:
            fov_sq_sum = (
                (lens1.fov - _nom('lens1.fov'))**2 +
                (lens2.fov - _nom('lens2.fov'))**2
            )
        breakdown['fov'] = reg['lambda_fov'] * fov_sq_sum
        breakdown['distortion'] = reg['lambda_distortion'] * (
            lens1.k1**2 + lens1.k2**2 + lens1.k3**2 +
            lens2.k1**2 + lens2.k2**2 + lens2.k3**2
        )
        breakdown['rotation'] = reg['lambda_rotation'] * (
            lens1.rotation_yaw**2 + lens1.rotation_pitch**2 + lens1.rotation_roll**2 +
            lens2.rotation_yaw**2 + lens2.rotation_pitch**2 + lens2.rotation_roll**2
        )

        return sum(breakdown.values()), breakdown
    
    def print_summary(self):
        print("\nOptimization Configuration:")
        print("-" * 50)
        opt_names = [name for name, _ in self.opt_params]
        print(f"Optimizing {len(opt_names)} parameters:")
        for name in opt_names:
            p = self.params[name]
            print(f"  {name}: [{p['min']:.4f}, {p['max']:.4f}] (nominal: {p['nominal']:.4f})")
        
        fixed = [n for n in self.PARAM_NAMES if n not in opt_names]
        if fixed:
            print(f"\nFixed ({len(fixed)}):")
            for name in fixed:
                p = self.params[name]
                link = " (=lens1)" if p.get('use_lens1') else ""
                print(f"  {name} = {p['nominal']:.4f}{link}")
        
        prep = self.preprocessing
        margin_pct = prep.get('edge_margin_ratio', 0.05) * 100
        norm_state = 'on' if prep.get('normalize_lighting', True) else 'off'
        vign_state = ('on (s=' + str(prep.get('vignette_strength', 1.0)) + ')'
                      if prep.get('vignette_correction', False) else 'off')
        mode = prep.get('comparison_mode', 'intensity')
        seam_overlap = float(prep.get('seam_overlap_deg', 0.0))
        seam_center = float(prep.get('seam_overlap_center_deg', 0.0))
        if seam_overlap > 0:
            ctr_label = (f"{seam_center:g} deg" if seam_center > 0 else "base_fov (default)")
            seam_state = f"{seam_overlap:g} deg wide, centred at FOV {ctr_label}"
        else:
            seam_state = "auto (full geometric overlap)"
        print(f"\nPreprocessing:"
              f"\n  normalize_lighting={norm_state}"
              f"\n  vignette_correction={vign_state}"
              f"\n  comparison_mode={mode}"
              f"\n  edge_margin={margin_pct:.0f}% of radius excluded"
              f"\n  seam strip = {seam_state}")
        
        reg = self.regularization
        if reg.get('enabled', True):
            print(f"Regularization: enabled")
            print(f"  lambda_center={reg['lambda_center']}, lambda_fov={reg['lambda_fov']}, "
                  f"lambda_dist={reg['lambda_distortion']}, lambda_rot={reg['lambda_rotation']}")
        else:
            print(f"Regularization: disabled")
        print("-" * 50)


def downsample_frames(frames_data, scale):
    if scale >= 1.0:
        return frames_data
    return [
        (cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
         cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        for l, r in frames_data
    ]


def equalize_vignette(img, n_bins: int = 32, strength: float = 1.0,
                      max_gain: float = 4.0):
    """Compensate for radial light fall-off in a fisheye image.

    Fisheye lenses follow approximately a cos^4(theta) brightness model: the
    OUTER EDGE of each lens circle is significantly darker than the centre.
    The seam between the two lenses sits at exactly that outer edge for both
    lenses, so a raw pixel-intensity comparison is dominated by vignetting
    rather than geometric misalignment.

    Algorithm:
      1. Bin the image into ``n_bins`` concentric annuli around the geometric
         centre (cx = w/2, cy = h/2).
      2. Compute the median brightness in each bin (robust to colourful pixels
         and saturated highlights).
      3. Build a per-radius gain that flattens the profile to the brightest
         inner bin, capped at ``max_gain`` so we don't amplify pure noise.
      4. Multiply the source pixels by that gain.

    Args:
        img:      H x W (gray) or H x W x 3 (BGR) image.  uint8 or float.
        n_bins:   Number of radial bins for the brightness profile.
        strength: 0.0 = no correction, 1.0 = fully flatten.  In between
                  blends linearly toward the original.
        max_gain: Maximum allowed multiplication (prevents amplifying noise
                  in the dimmest border pixels into white snow).

    Returns:
        Vignette-corrected image, same shape and dtype as input.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_r = min(w, h) / 2.0
    if max_r < 1:
        return img

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    yy, xx = np.indices(gray.shape, dtype=np.float32)
    r_norm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r
    bin_idx = np.clip((r_norm * n_bins).astype(np.int32), 0, n_bins - 1)

    valid_pixel = (gray > 5) & (gray < 250)
    profile = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        m = (bin_idx == b) & valid_pixel
        if np.sum(m) > 50:
            profile[b] = float(np.median(gray[m]))

    # Forward-fill bins with no data (e.g. the all-black border).
    last = profile[0] if profile[0] > 0 else 128.0
    for b in range(n_bins):
        if profile[b] <= 0:
            profile[b] = last
        else:
            last = profile[b]

    # Smooth with a 5-tap box filter to suppress per-bin noise.
    kernel = np.ones(5, dtype=np.float32) / 5.0
    profile = np.convolve(profile, kernel, mode='same')

    # Normalise to the brightest inner-quarter bin (a stable centre reference).
    centre = float(np.max(profile[: max(1, n_bins // 4)]))
    if centre <= 1.0:
        return img

    inv = centre / np.maximum(profile, 1.0)
    gain = 1.0 + strength * (inv - 1.0)
    gain = np.clip(gain, 1.0, max_gain).astype(np.float32)
    pix_gain = gain[bin_idx]

    if img.ndim == 3:
        out = img.astype(np.float32) * pix_gain[:, :, None]
    else:
        out = img.astype(np.float32) * pix_gain
    return np.clip(out, 0, 255).astype(img.dtype)


def preprocess_frames(frames_data, config):
    """Apply vignette correction and/or edge extraction to all frames once.

    Calling these on every objective evaluation would waste cycles -- the input
    fisheye frames don't change while the optimizer searches.  This runs them
    a single time before optimization and returns a new list of (left, right)
    pairs ready for ``calibrate_with_config``.

    Reads from ``config.preprocessing``:
        vignette_correction (bool)   -- enable equalize_vignette
        vignette_strength   (float)  -- strength argument for equalize_vignette
        comparison_mode     (str)    -- "intensity" (default) or "edges"
        edge_method         (str)    -- when comparison_mode = "edges":
                                        "sobel" (default), "scharr",
                                        "laplacian", or "canny"
    """
    prep = config.preprocessing
    do_vignette = bool(prep.get('vignette_correction', False))
    strength    = float(prep.get('vignette_strength', 1.0))
    mode        = str(prep.get('comparison_mode', 'intensity')).lower()
    edge_method = str(prep.get('edge_method', 'sobel')).lower()

    if not do_vignette and mode == 'intensity':
        return frames_data

    extras = f", edge_method={edge_method}" if mode == 'edges' else ''
    print(f"Preprocessing {len(frames_data)} frame(s): "
          f"vignette={'on (s=' + str(strength) + ')' if do_vignette else 'off'}, "
          f"comparison={mode}{extras}")

    out = []
    for left, right in frames_data:
        if do_vignette:
            left  = equalize_vignette(left,  strength=strength)
            right = equalize_vignette(right, strength=strength)
        if mode == 'edges':
            left  = extract_edges(left,  method=edge_method)
            right = extract_edges(right, method=edge_method)
        out.append((left, right))
    return out


def normalize_lighting(img):
    """Normalize lighting to focus on features rather than brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(img.shape) == 3 else img.astype(np.float32)
    valid = gray > 0
    if np.sum(valid) < 100:
        return img
    
    mean_val, std_val = np.mean(gray[valid]), max(np.std(gray[valid]), 1e-6)
    normalized = np.zeros_like(gray)
    normalized[valid] = np.clip(((gray[valid] - mean_val) / std_val) * 50 + 128, 0, 255)
    
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else normalized.astype(np.uint8)


def compute_seam_error(left_img, right_img, lens1, lens2, base_fov, normalize=True,
                       edge_margin_ratio: float = 0.05,
                       overlap_deg: float = 0.0,
                       overlap_center_deg: float = 0.0):
    """Compute seam alignment error using RMS difference on the seam strips.

    Projects each lens onto an equirectangular patch and compares the regions
    near the lens-to-lens boundary (the "seam"), one strip on each side of the
    panorama (front seam + back seam).

    Args:
        normalize: If True, normalize lighting (grayscale) before comparison.
        edge_margin_ratio: Fraction of the fisheye radius to exclude at the edge
                           (e.g. 0.05 removes the outermost 5% of each lens circle).
        overlap_deg: Total width of the seam comparison strip, in DEGREES OF FOV
                     (i.e. twice the half-angle from each boresight).  The strip
                     is centred on ``overlap_center_deg`` (see below).  Examples:
                       width=2.0, center=185 -> strip FOV [184, 186]
                       width=1.0, center=187 -> strip FOV [186.5, 187.5]
                     ``0`` -> use the full geometric overlap, i.e. strip FOV
                     [base_fov, lens_fov] (the legacy behaviour).
        overlap_center_deg: Centre of the strip, in degrees of FOV.  This must
                     be a FIXED value (independent of ``lens1.fov``/``lens2.fov``)
                     so that varying lens_fov during optimisation does NOT shift
                     which pixels are compared.  Default 0 -> ``base_fov`` (the
                     panorama hemisphere boundary, lon +/- 90 deg).
    """
    h, w = left_img.shape[:2]
    # NOTE: we no longer pre-mask with mask_fisheye_circle here.  That helper
    # uses the IMAGE centre as the disk centre, which is wrong when the lens
    # is offset (center_x != 0.5).  The fisheye_to_equirect_calibrated call
    # below now performs the disk-validity test using the LENS-CALIBRATION
    # centre and applies ``edge_margin_ratio`` directly, so output pixels
    # that were sampled from outside the original fisheye disk are correctly
    # marked invalid (and not counted in the comparison).
    right_flipped = np.fliplr(right_img)

    lens_fov = min(lens1.fov, lens2.fov)
    geo_overlap_deg = max(0.0, lens_fov - base_fov)

    # Decide the strip span in FOV coordinates.
    #
    # Convention: at longitude ``lon`` (deg), the "FOV equivalent" is ``2*|lon|``
    # i.e. the lens FOV needed to reach that longitude.  So FOV X <-> lon X/2.
    #
    # overlap_deg == 0 -> full geometric overlap, strip FOV [base_fov, lens_fov]
    # overlap_deg  > 0 -> strip FOV [center - width/2, center + width/2] where
    #                    center is independent of the dynamic lens_fov.
    if overlap_deg > 0.0:
        center_fov = overlap_center_deg if overlap_center_deg > 0.0 else base_fov
        strip_low_fov  = center_fov - overlap_deg / 2.0
        strip_high_fov = center_fov + overlap_deg / 2.0
    else:
        if geo_overlap_deg < 1.0:
            return 1000.0
        strip_low_fov  = base_fov
        strip_high_fov = lens_fov

    # FOV -> longitude (per-side).
    strip_low_lon  = strip_low_fov  / 2.0
    strip_high_lon = strip_high_fov / 2.0

    # Build the equirectangular projection wide enough to contain the upper end
    # of the strip.  Use the ``half_w / base_fov`` pix/deg scale so column
    # arithmetic stays simple.
    half_w = w
    out_h  = h
    proj_lens_fov = max(lens_fov, strip_high_fov)
    proj_w = int(round(half_w * proj_lens_fov / base_fov))

    try:
        left_patch, left_mask = fisheye_to_equirect_calibrated(
            left_img,      proj_w, out_h, lens1, proj_lens_fov,
            edge_margin_ratio=edge_margin_ratio)
        right_patch, right_mask = fisheye_to_equirect_calibrated(
            right_flipped, proj_w, out_h, lens2, proj_lens_fov,
            edge_margin_ratio=edge_margin_ratio)
    except Exception:
        return 1000.0

    right_patch, right_mask = np.fliplr(right_patch), np.fliplr(right_mask)

    # Convert the strip's longitudes to column indices in the projection.
    # Use floor on the low end and ceil on the high end so a sub-pixel-wide
    # strip still gets at least 1 pixel (otherwise low-resolution / very narrow
    # ``overlap_deg`` configs round to zero columns and we always return the
    # sentinel, leaving the optimizer with no gradient).
    # col(lon) = (lon + proj_lens_fov/2) * proj_w / proj_lens_fov
    def _lon_to_col_raw(lon):
        return (lon + proj_lens_fov / 2.0) * proj_w / proj_lens_fov

    front_low_col  = max(0,      int(math.floor(_lon_to_col_raw(strip_low_lon))))
    front_high_col = min(proj_w, int(math.ceil( _lon_to_col_raw(strip_high_lon))))
    if front_high_col <= front_low_col:
        # Strip falls entirely outside the projection — should not happen, but
        # guard anyway (callers treat 1000.0 as the no-overlap sentinel).
        return 1000.0

    overlap_px = front_high_col - front_low_col

    # ------------------------------------------------------------------
    # Per-pixel "missing-data" cost design.
    #
    # The seam strip lives in OUTPUT (equirect) space.  The optimiser's
    # parameters control which input pixels map there, so it can ARTIFICIALLY
    # invalidate strip pixels by:
    #   * shrinking lens.fov so r > 1 (out-of-FOV),
    #   * shifting center_x so the disk runs off the image rectangle,
    #   * inflating k1/k2 distortion so r is pushed outside [0, 1].
    # If invalid pixels were "free" (excluded from numerator AND denominator)
    # the optimiser would happily destroy the projection to escape any
    # hard-to-match pixel.  This is exactly what the user observed.
    #
    # Fix:
    #   * BOTH numerator and denominator are KEYED off the FIXED strip size.
    #     The optimiser cannot shrink the denominator.
    #   * one_black and both_black contribute a penalty equal to a worst-case
    #     photometric mismatch (BLACK_PEN per channel-pixel).  This means
    #     "make pixel invalid" is never cheaper than "leave pixel valid", so
    #     the optimiser is forced to keep coverage AND minimise photometric
    #     error simultaneously.
    #   * Pixels where the projection says "valid" but the value is
    #     essentially zero (the heavy Gear-360 vignette / black border)
    #     are demoted from "both_valid" to "missing data".  A zero-vs-zero
    #     match is not signal -- treating it as a perfect match is exactly
    #     what gave the optimiser a way to "win" by maximising black
    #     coverage in the strip.
    BLACK_PEN     = 128.0          # per-channel penalty for missing data
    BLACK_PEN_SQ  = BLACK_PEN ** 2
    DATA_THRESH   = 6              # 0..255 ; below this is treated as no-data

    total_sq_sum, total_count = 0.0, 0
    valid_count_diag = 0           # diagnostic: pixels with real both-side data
    one_black_count  = 0
    both_black_count = 0

    def _has_data(region, mask):
        """True where the projection was valid AND the pixel has real data
        (not the vignette-into-black at the rim of the input fisheye)."""
        if region.ndim == 3:
            bright = region.max(axis=-1)
        else:
            bright = region
        return mask & (bright > DATA_THRESH)

    def compute_region_error(r1, r2, mask1, mask2):
        """Accumulate squared differences; missing data carries BLACK_PEN."""
        nonlocal total_sq_sum, total_count
        nonlocal valid_count_diag, one_black_count, both_black_count

        has1 = _has_data(r1, mask1)
        has2 = _has_data(r2, mask2)

        both_valid = has1 & has2
        one_black  = (has1 ^ has2)         # exactly one side has data
        both_black = ~has1 & ~has2

        n_channels = 3 if len(r1.shape) == 3 else 1

        # 1. Both valid -> real photometric RMS over real data.
        if np.any(both_valid):
            r1_proc = normalize_lighting(r1) if normalize else r1
            r2_proc = normalize_lighting(r2) if normalize else r2

            if n_channels == 3:
                valid_3d = np.broadcast_to(both_valid[..., None],
                                           r1_proc.shape)
                diff_sq = np.where(
                    valid_3d,
                    (r1_proc.astype(np.float32) - r2_proc.astype(np.float32)) ** 2,
                    0.0,
                )
                total_sq_sum += float(np.sum(diff_sq))
                total_count += int(np.sum(both_valid)) * n_channels
            else:
                diff_sq = np.where(
                    both_valid,
                    (r1_proc.astype(np.float32) - r2_proc.astype(np.float32)) ** 2,
                    0.0,
                )
                total_sq_sum += float(np.sum(diff_sq))
                total_count += int(np.sum(both_valid))

        # 2. One side missing -> constant max-mismatch penalty per channel.
        #    Counted in BOTH numerator AND denominator so the strip-pixel
        #    budget is fixed.
        n_one = int(np.sum(one_black))
        if n_one > 0:
            total_sq_sum += n_one * n_channels * BLACK_PEN_SQ
            total_count += n_one * n_channels

        # 3. Both sides missing -> same max-mismatch penalty per channel.
        n_both = int(np.sum(both_black))
        if n_both > 0:
            total_sq_sum += n_both * n_channels * BLACK_PEN_SQ
            total_count += n_both * n_channels

        valid_count_diag += int(np.sum(both_valid)) * n_channels
        one_black_count  += n_one * n_channels
        both_black_count += n_both * n_channels

    # Front seam: left columns [front_low_col, front_high_col) match the
    # mirror columns of the (already-fliplr'd) right patch.
    mirror_low  = proj_w - front_high_col
    mirror_high = proj_w - front_low_col

    r1 = left_patch[:, front_low_col:front_high_col]
    r2 = right_patch[:, mirror_low:mirror_high]
    m1 = left_mask[:,  front_low_col:front_high_col]
    m2 = right_mask[:, mirror_low:mirror_high]
    compute_region_error(r1, r2, m1, m2)

    # Back seam: column ranges swap between left and right.
    r1 = left_patch[:, mirror_low:mirror_high]
    r2 = right_patch[:, front_low_col:front_high_col]
    m1 = left_mask[:,  mirror_low:mirror_high]
    m2 = right_mask[:, front_low_col:front_high_col]
    compute_region_error(r1, r2, m1, m2)

    if total_count <= 0:
        # Strip slice was zero-sized for some reason; report the worst case.
        return BLACK_PEN
    return float(np.sqrt(total_sq_sum / total_count))


def evaluate_objective(params, frames_data, base_fov, config):
    """Return (total, seam_rms_mean, reg_penalty, reg_breakdown).

    ``seam_rms_mean`` is the arithmetic mean of ``compute_seam_error`` over every
    (left, right) pair in ``frames_data`` — i.e. mean over all frames used.
    """
    lens1, lens2 = config.vector_to_lenses(params)
    normalize = config.preprocessing.get('normalize_lighting', True)
    edge_margin = config.preprocessing.get('edge_margin_ratio', 0.05)
    overlap_deg = float(config.preprocessing.get('seam_overlap_deg', 0.0))
    overlap_center_deg = float(config.preprocessing.get('seam_overlap_center_deg', 0.0))
    n = len(frames_data)
    if n <= 0:
        raise ValueError("frames_data is empty")
    seam_mean = sum(
        compute_seam_error(l, r, lens1, lens2, base_fov,
                           normalize=normalize,
                           edge_margin_ratio=edge_margin,
                           overlap_deg=overlap_deg,
                           overlap_center_deg=overlap_center_deg)
        for l, r in frames_data
    ) / n
    reg_penalty, reg_breakdown = config.compute_regularization(lens1, lens2)
    return seam_mean + reg_penalty, seam_mean, reg_penalty, reg_breakdown


def objective_with_config(params, frames_data, base_fov, config):
    """Objective: E = mean(seam RMS over frames) + regularization."""
    total, _, _, _ = evaluate_objective(params, frames_data, base_fov, config)
    return total


def _make_preview_image(left_img, right_img, lens1, lens2, base_fov,
                         max_width: int = 1280):
    """Stitch a quick equirectangular preview from a single (left, right) pair
    using the current parameter vector.

    Returned image is BGR uint8, downscaled to fit ``max_width`` pixels wide
    so it stays responsive even when the source frames are large.
    """
    calib = CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)
    try:
        out, _, _, _ = stitch_dual_fisheye(
            left_img, right_img, calib, base_fov, blend=False)
    except Exception as exc:
        # Build a black "broken" frame so the window stays alive.
        h = left_img.shape[0]
        w = left_img.shape[1] * 2
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(out, f"projection error: {exc}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if out.shape[1] > max_width:
        scale = max_width / out.shape[1]
        nw = int(out.shape[1] * scale)
        nh = int(out.shape[0] * scale)
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _annotate_preview(img, *, gen: int, total: float, seam: float,
                      lens1, lens2):
    """Overlay live optimisation status on the preview image (in place)."""
    lines = [
        f"gen {gen}   obj={total:.2f}   seam={seam:.2f}",
        f"L1 c=({lens1.center_x:.4f},{lens1.center_y:.4f}) "
        f"fov={lens1.fov:.2f} k1={lens1.k1:+.4f}",
        f"L2 c=({lens2.center_x:.4f},{lens2.center_y:.4f}) "
        f"fov={lens2.fov:.2f} k1={lens2.k1:+.4f}",
    ]
    h = img.shape[0]
    box_h = 22 * len(lines) + 10
    cv2.rectangle(img, (0, h - box_h), (img.shape[1], h),
                  (0, 0, 0), thickness=cv2.FILLED)
    for i, text in enumerate(lines):
        y = h - box_h + 22 * (i + 1)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)


def calibrate_with_config(frames_data, base_fov, config, workers=1,
                          preview: bool = False,
                          preview_window: str = "calibration preview"):
    """Optimization using differential evolution with TOML config.

    Args:
        preview: If True, opens an OpenCV window and updates a stitched
                 equirectangular preview after every generation.  Uses the
                 first frame in ``frames_data`` (the same frame the solver
                 sees for the leading sample).
    """
    # ---- Geometry sanity check (run BEFORE any heavy work) -------------------
    # The seam optimizer aligns the OVERLAP region between the two lenses.
    # That overlap exists only if the physical lens FOV exceeds the equirect
    # base FOV.  If lens.fov <= base_fov the seam error degenerates to a
    # constant sentinel (1000) and the optimizer cannot learn anything.
    # Fail fast with a clear diagnostic instead of running for hours producing
    # garbage.
    nominal_lens_fov = min(
        config.params['lens1.fov']['nominal'],
        config.params['lens2.fov']['nominal'],
    )
    if nominal_lens_fov <= base_fov + 0.5:
        raise ValueError(
            f"Geometry invalid: smaller nominal lens FOV ({nominal_lens_fov:.1f} deg) "
            f"is not greater than the projection base FOV ({base_fov:.1f} deg).\n"
            "  The two lenses can only be aligned through their OVERLAP, which\n"
            "  requires lens.fov > base_fov by at least a few degrees.\n"
            "  Fixes:\n"
            f"    - Raise [lens1.fov].nominal in adjoint_tuning.toml above {base_fov:.0f} "
            "(Gear 360 lenses are physically ~195 deg).\n"
            "    - Or lower --fov on the command line "
            "(should normally be 180 -- a hemisphere)."
        )

    print(f"\n{'='*60}")
    print("CALIBRATION - Differential Evolution")
    print(f"{'='*60}")
    config.print_summary()

    # Also warn (not fatal) if the OPTIMIZED lower bound dips into no-overlap.
    for fov_param in ('lens1.fov', 'lens2.fov'):
        p = config.params[fov_param]
        if p['optimize'] and p['min'] <= base_fov:
            print(f"Warning: {fov_param}.min ({p['min']}) <= base_fov ({base_fov}); "
                  f"the optimizer may waste evaluations in the no-overlap region.")

    # Warn if the (custom) seam strip is too narrow for the current resolution.
    # The strip spans ``seam_overlap_deg`` degrees of FOV, sampled at the
    # equirectangular pixel scale of ``half_w / base_fov`` pix/deg.  If that
    # is < 1 px the strip rounds to a single column of arbitrary phase and
    # the seam RMS becomes nearly constant.
    seam_overlap_deg = float(config.preprocessing.get('seam_overlap_deg', 0.0))
    if seam_overlap_deg > 0.0 and frames_data:
        sample = frames_data[0][0]  # first left-half image
        half_w = sample.shape[1]
        # Strip width in pixels at scale half_w pix per base_fov deg.
        # FOV-degrees -> longitude-degrees: divide by 2.
        strip_px = (seam_overlap_deg / 2.0) * (half_w / base_fov)
        if strip_px < 1.0:
            need_deg = math.ceil(2.0 * base_fov / max(1, half_w) * 10) / 10
            print(f"Warning: seam_overlap_deg={seam_overlap_deg} deg is sub-pixel at "
                  f"the current resolution ({half_w} px/half, ~{strip_px:.2f} px wide).\n"
                  f"  The strip is forced to 1 column, which gives a near-constant\n"
                  f"  seam RMS and the optimizer may not converge.\n"
                  f"  Fixes:\n"
                  f"    - Raise --seam-overlap-deg above ~{need_deg:g} for this resolution, OR\n"
                  f"    - Run with a larger --scale (e.g. --scale 0.5 or 1.0).")

    # Capture the FIRST frame BEFORE preprocessing so the preview shows the
    # actual stitched output (not vignette-flattened or edge-magnitude data).
    preview_pair = None
    preview_enabled = bool(preview) and len(frames_data) > 0
    if preview_enabled:
        preview_pair = (frames_data[0][0].copy(), frames_data[0][1].copy())

    # Apply vignette / edge preprocessing once before optimization.
    # In 'edges' mode the global lighting normalisation is redundant
    # (gradient magnitudes are already brightness-invariant), so disable it.
    if str(config.preprocessing.get('comparison_mode', 'intensity')).lower() == 'edges':
        if config.preprocessing.get('normalize_lighting', True):
            print("Note: comparison_mode=edges -> auto-disabling normalize_lighting (redundant)")
            config.preprocessing['normalize_lighting'] = False
    frames_data = preprocess_frames(frames_data, config)
    n_frames_used = len(frames_data)

    bounds = config.get_bounds()
    n_params = config.get_n_params()
    maxiter = config.optimizer.get('maxiter', 100)
    popsize = config.optimizer.get('popsize', 15)
    strategy = config.optimizer.get('strategy', 'best1bin')
    tol = config.optimizer.get('tol', 0.01)
    mutation = config.optimizer.get('mutation', [0.5, 1.0])
    recombination = config.optimizer.get('recombination', 0.7)
    if isinstance(mutation, list):
        mutation = tuple(mutation)

    print(f"\nPopulation: {popsize * n_params}, Max generations: {maxiter}\n")

    init_vector = config.get_initial_vector()
    _, init_seam_mean, _, _ = evaluate_objective(
        init_vector, frames_data, base_fov, config)
    print(f"Initial seam RMS (mean over {n_frames_used} frame(s)): {init_seam_mean:.2f}")
    if init_seam_mean >= 999.0:
        print("Warning: initial seam error hit the no-overlap sentinel (1000) -- "
              "check that lens.fov bounds straddle a region where lens.fov > base_fov.")
    print()

    # ---- Live preview window ------------------------------------------------
    # Open lazily so headless / no-GUI environments don't crash if the user
    # didn't ask for it.  We disable the preview if cv2 cannot create a window.
    def _preview_render(xk, *, gen: int, total: float, seam: float):
        if not preview_enabled or preview_pair is None:
            return
        l1, l2 = config.vector_to_lenses(xk)
        img = _make_preview_image(preview_pair[0], preview_pair[1],
                                  l1, l2, base_fov)
        _annotate_preview(img, gen=gen, total=total, seam=seam,
                          lens1=l1, lens2=l2)
        try:
            cv2.imshow(preview_window, img)
            cv2.waitKey(1)  # pump GUI events; non-blocking
        except cv2.error:
            pass  # GUI not available (e.g. headless build of opencv)

    if preview_enabled:
        try:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window, 1024, 512)
            print(f"Live preview enabled -- updates after every generation "
                  f"(window: '{preview_window}'). Close from this terminal "
                  f"with Ctrl-C.")
            _preview_render(init_vector, gen=0, total=init_seam_mean,
                            seam=init_seam_mean)
        except cv2.error as exc:
            print(f"Note: preview disabled -- OpenCV GUI unavailable ({exc})")
            preview_enabled = False

    best_so_far, gen_count = [float('inf')], [0]
    
    def callback(xk, convergence):
        gen_count[0] += 1
        total, seam_mean, _, _ = evaluate_objective(xk, frames_data, base_fov, config)
        marker = " (new best)" if total < best_so_far[0] else ""
        if total < best_so_far[0]:
            best_so_far[0] = total
        print(f"  gen {gen_count[0]:3d}: obj={total:.2f}  "
              f"seam_mean({n_frames_used} frames)={seam_mean:.2f}{marker}")
        if preview_enabled:
            _preview_render(xk, gen=gen_count[0], total=total, seam=seam_mean)
        return False
    
    result = differential_evolution(
        objective_with_config, bounds=bounds, args=(frames_data, base_fov, config),
        strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol,
        mutation=mutation, recombination=recombination, seed=42,
        callback=callback, polish=True, updating='deferred', workers=workers
    )
    
    print(f"\n  final objective (seam_mean + reg): {result.fun:.2f}")
    print(f"  {result.message}, {result.nfev} evaluations")
    
    _, seam_error, reg_penalty, reg = evaluate_objective(
        result.x, frames_data, base_fov, config)
    lens1, lens2 = config.vector_to_lenses(result.x)

    print(f"\n{'='*60}")
    print("OBJECTIVE BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Seam RMS (mean over {n_frames_used} frame(s)): {seam_error:.2f}")
    print(f"  Regularization: {reg_penalty:.2f}")
    print(f"    center={reg['center']:.4f}, fov={reg['fov']:.4f}, dist={reg['distortion']:.4f}, rot={reg['rotation']:.4f}")
    print(f"  Total:        {result.fun:.2f}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZED PARAMETERS")
    print(f"{'='*60}")
    for i, lens in enumerate([lens1, lens2], 1):
        print(f"\nLens {i}:")
        print(f"  center: ({lens.center_x:.6f}, {lens.center_y:.6f})")
        print(f"  fov: {lens.fov:.2f}°")
        print(f"  k: [{lens.k1:.6f}, {lens.k2:.6f}, {lens.k3:.6f}]")
        print(f"  rot: [{lens.rotation_yaw:.6f}, {lens.rotation_pitch:.6f}, {lens.rotation_roll:.6f}]")

    # Show the FINAL preview and wait briefly for the user to inspect.
    if preview_enabled:
        try:
            _preview_render(result.x, gen=gen_count[0],
                            total=result.fun, seam=seam_error)
            print("Preview window is showing the final stitch. "
                  "Press any key in the window to close.")
            cv2.waitKey(0)
        except cv2.error:
            pass
        finally:
            try:
                cv2.destroyWindow(preview_window)
            except cv2.error:
                pass

    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def project_equirectangular(left_img, right_img, calib, base_fov, blend=False):
    """Project dual-fisheye to equirectangular."""
    left_masked, _ = mask_fisheye_circle(left_img)
    right_masked, _ = mask_fisheye_circle(right_img)
    result, _, _, _ = stitch_dual_fisheye(left_masked, right_masked, calib, base_fov, blend=blend)
    return result


def main():
    parser = argparse.ArgumentParser(description='Dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Dual-fisheye image')
    parser.add_argument('--video', help='Video for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--config', '-c', help='TOML config file')
    parser.add_argument('--fov', type=float, default=180.0, help='Base FOV (degrees)')
    parser.add_argument('--frames', type=int, default=None, help='Upper limit of first N frames to consider (default: all)')
    parser.add_argument('--frame-rate', type=float, default=1.0, help='Frame sampling rate (0.2 = every 5th frame, 0.5 = every 2nd)')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--maxiter', type=int, help='Override max generations')
    parser.add_argument('--popsize', type=int, help='Override population size')
    parser.add_argument('--no-regularization', action='store_true', help='Disable regularization (pure seam error)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable lighting normalization (use raw RGB)')
    parser.add_argument('--no-vignette', action='store_true',
                        help='Disable vignette equalization (default: enabled)')
    parser.add_argument('--vignette-strength', type=float, default=None,
                        help='Override vignette correction strength (0.0..1.0)')
    parser.add_argument('--comparison-mode', choices=['intensity', 'edges'], default=None,
                        help='Override comparison_mode: intensity (raw pixels) or edges (Sobel)')
    parser.add_argument('--seam-overlap-deg', type=float, default=None,
                        help='Total seam-strip width in degrees of FOV.  '
                             '0 = use the full geometric overlap (legacy).  '
                             'Use with --seam-overlap-center-deg to anchor the strip.')
    parser.add_argument('--seam-overlap-center-deg', type=float, default=None,
                        help='Centre of the seam-strip in degrees of FOV (FIXED, NOT '
                             'tied to lens.fov).  0 = base_fov (180 deg).  '
                             'E.g. width=2.0, center=185 -> strip FOV [184, 186].')
    parser.add_argument('--output-video', help='Output equirectangular video path (requires --video)')
    parser.add_argument('--output-frame-rate', type=float, default=1.0, help='Output video frame rate (0.5 = every 2nd frame)')
    parser.add_argument('--save-json', metavar='PATH', help='Also write the raw calibration JSON to PATH')
    parser.add_argument('--preview', action='store_true',
                        help='Open a live OpenCV window that shows the stitched '
                             'first frame and refreshes after every generation.')
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("Provide --image or --video")

    # If the user passed --config explicitly, require it to exist (no silent fallback).
    # If they didn't, look for the canonical adjoint_tuning.toml; if even that is missing,
    # fall back to compiled defaults.
    if args.config:
        config = OptimizationConfig(args.config, require_config=True)
    elif DEFAULT_CONFIG_PATH.exists():
        config = OptimizationConfig(str(DEFAULT_CONFIG_PATH))
    else:
        print(f"Note: {DEFAULT_CONFIG_PATH} not found -- using built-in defaults")
        config = OptimizationConfig(None)
    
    if args.maxiter:
        config.optimizer['maxiter'] = args.maxiter
    if args.popsize:
        config.optimizer['popsize'] = args.popsize
    if args.no_regularization:
        config.regularization['enabled'] = False
    if args.no_normalize:
        config.preprocessing['normalize_lighting'] = False
    if args.no_vignette:
        config.preprocessing['vignette_correction'] = False
    if args.vignette_strength is not None:
        config.preprocessing['vignette_strength'] = args.vignette_strength
    if args.comparison_mode is not None:
        config.preprocessing['comparison_mode'] = args.comparison_mode
    if args.seam_overlap_deg is not None:
        config.preprocessing['seam_overlap_deg'] = float(args.seam_overlap_deg)
    if args.seam_overlap_center_deg is not None:
        config.preprocessing['seam_overlap_center_deg'] = float(args.seam_overlap_center_deg)

    
    frames_data, source_frame = [], None
    
    if args.image:
        print(f"Loading: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Error: Cannot load {args.image}")
        h, w = img.shape[:2]
        left, right = img[:, :w//2], img[:, w//2:]
        frames_data, source_frame = [(left, right)], (left, right)
    else:
        print(f"Loading video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Error: Cannot open {args.video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, first = cap.read()
        if ret:
            h, w = first.shape[:2]
            source_frame = (first[:, :w//2], first[:, w//2:])
        
        # --frames: upper limit of first N frames to consider (default: all)
        # --frame-rate: sampling rate (0.2 = every 5th frame)
        max_frame = min(args.frames, total) if args.frames else total
        step = max(1, int(1.0 / args.frame_rate)) if args.frame_rate > 0 else 1
        
        frame_indices = list(range(0, max_frame, step))
        print(f"  Total frames: {total}, considering first {max_frame}, sampling every {step} (rate={args.frame_rate})")
        print(f"  Using {len(frame_indices)} frames: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                frames_data.append((frame[:, :w//2], frame[:, w//2:]))
        cap.release()
    
    print(f"Using {len(frames_data)} frames, FOV={args.fov}°")
    
    if args.scale < 1.0:
        h, w = frames_data[0][0].shape[:2]
        print(f"Downsampling to {int(w*args.scale)}x{int(h*args.scale)}")
        frames_data = downsample_frames(frames_data, args.scale)
    
    calib = calibrate_with_config(frames_data, args.fov, config,
                                   workers=args.workers, preview=args.preview)

    if args.save_json:
        calib.save_json(args.save_json)
        print(f"\nSaved JSON: {args.save_json}")

    if args.output and source_frame:
        result = project_equirectangular(source_frame[0], source_frame[1], calib, args.fov, blend=False)
        cv2.imwrite(args.output, result)
        print(f"Saved: {args.output}")
    
    # Generate output video if requested
    if args.output_video and args.video:
        from camera_calibration.projections.fisheye_to_equirectangular import process_video
        process_video(
            args.video, args.output_video, calib, args.fov,
            blend=True, frame_rate=args.output_frame_rate
        )


if __name__ == '__main__':
    main()
