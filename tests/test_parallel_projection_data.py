"""
Integration tests that exercise CTSlice and CTRadon on the provided
parallel-projection sinograms (SheppLoganPhantom, Lotus, Walnut).

The goal is to ensure both reconstruction methods can process the real
datasets and produce broadly comparable results for both 180° and 360°
acquisition ranges. The absolute thresholds were measured empirically
and act as regression guards instead of strict mathematical guarantees.
"""

from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pytest
from skimage.transform import resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Parallel Projection"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_slice import CTRadon, CTSlice  # noqa: E402  (import after sys.path tweak)


def _resolve_image_reader() -> Callable[[Path], np.ndarray]:
    """
    Provide an image reader that works whether or not OpenCV is available.
    """

    try:
        from imageio.v2 import imread  # type: ignore

        def _reader(path: Path) -> np.ndarray:
            return np.asarray(imread(path), dtype=np.float64)

        return _reader
    except ImportError:  # pragma: no cover
        import cv2

        def _reader(path: Path) -> np.ndarray:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not load sinogram: {path}")
            return image.astype(np.float64)

        return _reader


READ_IMAGE = _resolve_image_reader()


def _load_parallel_sinogram(filename: str) -> np.ndarray:
    """
    Load and normalize a provided parallel sinogram to the [0, 1] range.
    """

    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Parallel sinogram not available: {path}")

    sinogram = READ_IMAGE(path)
    if sinogram.ndim == 3:
        sinogram = sinogram[..., 0]

    if sinogram.max() > 0:
        sinogram = sinogram / 255.0

    return sinogram.astype(np.float64, copy=False)


def _normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0, 1] without mutating the input.
    """

    normalized = image - np.min(image)
    max_val = np.max(normalized)
    if max_val > 0:
        normalized = normalized / max_val
    return normalized


def _align_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize the provided image if necessary so both reconstructions share a grid.
    """

    if image.shape == target_shape:
        return image
    return resize(
        image,
        target_shape,
        preserve_range=True,
        anti_aliasing=False,
    )


def _normalized_mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the MAE after independently normalizing both images to [0, 1].
    """

    norm_a = _normalize(a)
    norm_b = _normalize(b)
    return float(np.mean(np.abs(norm_a - norm_b)))


@pytest.mark.parametrize(
    ("filename", "angle_range", "max_mae"),
    [
        ("SheppLoganPhantom.png", 180, 0.15),
        ("Lotus.png", 360, 0.45),
        ("Walnut.png", 360, 0.30),
    ],
)
def test_parallel_sinograms_ctslice_vs_ctradon(filename: str, angle_range: int, max_mae: float) -> None:
    """
    Ensure CTSlice and CTRadon stay broadly in agreement on provided datasets.

    The thresholds allow for existing numerical differences yet will detect
    future regressions that would increase the gap between both methods.
    """

    sinogram = _load_parallel_sinogram(filename)
    reconstruction_dfr = CTSlice(sinogram, angle_range=angle_range)
    reconstruction_fbp = CTRadon(sinogram, angle_range=angle_range)

    assert np.isfinite(reconstruction_dfr).all(), "CTSlice produced non-finite values"
    assert np.isfinite(reconstruction_fbp).all(), "CTRadon produced non-finite values"

    reconstruction_dfr = _align_shape(reconstruction_dfr, reconstruction_fbp.shape)
    mae = _normalized_mae(reconstruction_dfr, reconstruction_fbp)

    assert mae <= max_mae, (
        f"Normalized MAE {mae:.3f} exceeded {max_mae:.3f} "
        f"for dataset {filename} at {angle_range}°"
    )

