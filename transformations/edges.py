"""Image-edge extraction for fisheye seam alignment.

The seam-error optimizer compares the OVERLAP between the two lenses.
Comparing raw pixel intensities conflates geometric misregistration with
brightness / exposure differences between the lenses; comparing GRADIENT
MAGNITUDES removes the brightness component and lets the optimizer focus on
structural alignment.

This module exposes a single function, :func:`extract_edges`, with a small
choice of operators.  It is intentionally dependency-light: only numpy and
OpenCV are required.
"""

from __future__ import annotations

import cv2
import numpy as np


_VALID_METHODS = ("sobel", "scharr", "laplacian", "canny")


def extract_edges(
    img: np.ndarray,
    method: str = "sobel",
    blur_ksize: int = 10,
    normalize: bool = True,
    canny_lo: int = 0,
    canny_hi: int = 255,
) -> np.ndarray:
    """Convert an image into a single-channel edge / gradient-magnitude map.

    Args:
        img:         H x W or H x W x 3 image (uint8 or float).  Colour input
                     is converted to grayscale first.
        method:      Edge operator -- one of:

                       * ``"sobel"``     -- Sobel x/y, sqrt(gx^2 + gy^2). Default.
                                            Smooth, continuous, good for
                                            optimization signals.
                       * ``"scharr"``    -- Scharr operator (more accurate than
                                            Sobel at small kernel sizes).
                       * ``"laplacian"`` -- |Laplacian|, rotationally isotropic.
                       * ``"canny"``     -- Canny edge detector (binary, sparse).
                                            Use for very clean structural matches.

        blur_ksize:  Gaussian pre-blur kernel size to suppress sensor noise.
                     0 or 1 disables blur.  Even values are bumped to the next
                     odd value (OpenCV requirement).
        normalize:   Rescale the magnitude image to the full 0..255 range.
                     Automatically off for ``method="canny"`` (already binary).
        canny_lo:    Lower hysteresis threshold for Canny (only when method="canny").
        canny_hi:    Upper hysteresis threshold for Canny (only when method="canny").

    Returns:
        Single-channel uint8 image with shape (H, W).
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS}, got {method!r}"
        )

    # ---- Coerce to a dtype OpenCV's filters accept ----
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype not in (np.uint8, np.float32):
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # ---- Optional blur ----
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # ---- Edge operator ----
    if method == "canny":
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return cv2.Canny(gray, canny_lo, canny_hi)

    if method == "laplacian":
        mag = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    elif method == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
    else:  # sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

    if normalize:
        peak = float(mag.max())
        if peak > 1e-6:
            mag = mag * (255.0 / peak)
    return np.clip(mag, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Standalone edge-extraction CLI.

    Examples:
      python edges.py -i frame.jpg -o edges.png
      python edges.py -i frame.jpg -o edges.png --method canny --canny-lo 80 --canny-hi 200
      python edges.py -i frame.jpg -o edges.png --method scharr --blur 5
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Extract edges / gradient magnitudes from an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE",
                        help="Input image (any OpenCV-readable format).")
    parser.add_argument("-o", "--output", required=True, metavar="FILE",
                        help="Output image path (PNG / JPG).")
    parser.add_argument("-m", "--method", choices=_VALID_METHODS, default="sobel",
                        help="Edge operator (default: sobel).")
    parser.add_argument("--blur", type=int, default=3, metavar="K",
                        help="Pre-blur Gaussian kernel size; 0 or 1 disables (default: 3).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Do NOT rescale the output to 0..255.")
    parser.add_argument("--canny-lo", type=int, default=50,
                        help="Canny lower hysteresis threshold (default: 50).")
    parser.add_argument("--canny-hi", type=int, default=150,
                        help="Canny upper hysteresis threshold (default: 150).")
    parser.add_argument("--show", action="store_true",
                        help="Also display input + output side-by-side via OpenCV.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read image: {in_path}")

    edges = extract_edges(
        img,
        method=args.method,
        blur_ksize=args.blur,
        normalize=not args.no_normalize,
        canny_lo=args.canny_lo,
        canny_hi=args.canny_hi,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), edges):
        sys.exit(f"Failed to write output: {out_path}")

    print(f"Edges ({args.method}) saved: {out_path}  "
          f"shape={edges.shape}, dtype={edges.dtype}, "
          f"min={int(edges.min())}, max={int(edges.max())}")

    if args.show:
        gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img.ndim == 3 else img)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        if gray.shape != edges.shape:
            gray = cv2.resize(gray, (edges.shape[1], edges.shape[0]))
        side_by_side = np.hstack([gray, edges])
        cv2.imshow("input | edges (q to quit)", side_by_side)
        while True:
            if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
