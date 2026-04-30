"""
Calibration configuration structures for fisheye camera.
"""

from dataclasses import dataclass, asdict
import json
import math
from typing import Optional
import numpy as np


@dataclass
class LensCalibration:
    """Per-lens calibration parameters."""
    center_x: float = 0.5  # Normalized center X (0-1)
    center_y: float = 0.5  # Normalized center Y (0-1)
    fov: float = 180.0  # Field of view in degrees (actual lens FOV)
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3
    rotation_yaw: float = 0.0  # Rotation in radians
    rotation_pitch: float = 0.0
    rotation_roll: float = 0.0


@dataclass
class CameraCalibration:
    """Full dual-lens camera calibration."""
    lens1: LensCalibration
    lens2: LensCalibration
    is_horizontal: bool = True  # Horizontal (True) or vertical (False) layout
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'lens1CenterX': self.lens1.center_x,
            'lens1CenterY': self.lens1.center_y,
            'lens1FOV': self.lens1.fov,
            'lens1K1': self.lens1.k1,
            'lens1K2': self.lens1.k2,
            'lens1K3': self.lens1.k3,
            'lens1RotationYaw': self.lens1.rotation_yaw,
            'lens1RotationPitch': self.lens1.rotation_pitch,
            'lens1RotationRoll': self.lens1.rotation_roll,
            'lens2CenterX': self.lens2.center_x,
            'lens2CenterY': self.lens2.center_y,
            'lens2FOV': self.lens2.fov,
            'lens2K1': self.lens2.k1,
            'lens2K2': self.lens2.k2,
            'lens2K3': self.lens2.k3,
            'lens2RotationYaw': self.lens2.rotation_yaw,
            'lens2RotationPitch': self.lens2.rotation_pitch,
            'lens2RotationRoll': self.lens2.rotation_roll,
            'isHorizontal': self.is_horizontal,
        }
    
    @classmethod
    def from_dict(cls, d):
        """Create from dictionary (JSON format)."""
        # Handle FOV: new format uses degrees (>10), legacy used scale factor (<10)
        def parse_fov(val, default=180.0):
            if val is None:
                return default
            # Legacy scale factors are typically 0.9-1.2, new FOV values are 150-220
            if val < 10:
                # Legacy scale factor - convert assuming 180° base
                return 180.0 * val
            return val
        
        lens1_fov = d.get('lens1FOV', d.get('lensFOV', 180.0))
        lens2_fov = d.get('lens2FOV', d.get('lensFOV', 180.0))
        
        lens1 = LensCalibration(
            center_x=d.get('lens1CenterX', 0.5),
            center_y=d.get('lens1CenterY', 0.5),
            fov=parse_fov(lens1_fov),
            k1=d.get('lens1K1', 0.0),
            k2=d.get('lens1K2', 0.0),
            k3=d.get('lens1K3', 0.0),
            rotation_yaw=d.get('lens1RotationYaw', d.get('lensRotationYaw', 0.0)),
            rotation_pitch=d.get('lens1RotationPitch', d.get('lensRotationPitch', 0.0)),
            rotation_roll=d.get('lens1RotationRoll', d.get('lensRotationRoll', 0.0)),
        )
        lens2 = LensCalibration(
            center_x=d.get('lens2CenterX', 0.5),
            center_y=d.get('lens2CenterY', 0.5),
            fov=parse_fov(lens2_fov),
            k1=d.get('lens2K1', 0.0),
            k2=d.get('lens2K2', 0.0),
            k3=d.get('lens2K3', 0.0),
            rotation_yaw=d.get('lens2RotationYaw', d.get('lensRotationYaw', 0.0)),
            rotation_pitch=d.get('lens2RotationPitch', d.get('lensRotationPitch', 0.0)),
            rotation_roll=d.get('lens2RotationRoll', d.get('lensRotationRoll', 0.0)),
        )
        return cls(lens1=lens1, lens2=lens2, is_horizontal=d.get('isHorizontal', True))
    
    @classmethod
    def load_json(cls, filepath):
        """Load calibration from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
            params = data.get('parameters', data)
            return cls.from_dict(params)

    @classmethod
    def load_toml(cls, filepath):
        """Load calibration from TOML file (cpp/calibration.toml format).

        TOML format differences from the internal JSON format:
          - fov is a scale factor (1.0 = 180 deg), converted to degrees here.
          - distortion uses p1/p2/p3 keys (mapped to k1/k2/k3).
          - rotation_yaw/pitch/roll are stored in degrees, converted to radians.
        """
        try:
            import tomllib
            open_fn = lambda p: open(p, 'rb')
            load_fn = tomllib.load
        except ImportError:
            try:
                import toml as tomllib  # type: ignore
                open_fn = lambda p: open(p, 'r', encoding='utf-8')
                load_fn = tomllib.load
            except ImportError:
                raise ImportError("tomllib (Python 3.11+) or the 'toml' package is required to load TOML files.")

        with open_fn(filepath) as f:
            data = load_fn(f)

        def _parse_lens(section):
            fov_raw = section.get('fov', 1.0)
            fov_deg = fov_raw * 180.0 if fov_raw < 10.0 else fov_raw
            return LensCalibration(
                center_x=section.get('center_x', 0.5),
                center_y=section.get('center_y', 0.5),
                fov=fov_deg,
                k1=section.get('p1', section.get('k1', 0.0)),
                k2=section.get('p2', section.get('k2', 0.0)),
                k3=section.get('p3', section.get('k3', 0.0)),
                rotation_yaw=math.radians(section.get('rotation_yaw', 0.0)),
                rotation_pitch=math.radians(section.get('rotation_pitch', 0.0)),
                rotation_roll=math.radians(section.get('rotation_roll', 0.0)),
            )

        lens1 = _parse_lens(data.get('lens1', {}))
        lens2 = _parse_lens(data.get('lens2', {}))
        return cls(lens1=lens1, lens2=lens2)

    @classmethod
    def load(cls, filepath):
        """Load calibration from JSON or TOML file (auto-detected by extension)."""
        p = str(filepath)
        if p.lower().endswith('.toml'):
            return cls.load_toml(filepath)
        return cls.load_json(filepath)

    def save_json(self, filepath):
        """Save calibration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({'parameters': self.to_dict()}, f, indent=2)


def create_default_calibration():
    """Create default calibration (no corrections)."""
    lens1 = LensCalibration()
    lens2 = LensCalibration()
    return CameraCalibration(lens1=lens1, lens2=lens2)
