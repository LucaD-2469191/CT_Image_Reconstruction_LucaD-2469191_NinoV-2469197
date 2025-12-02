import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_slice import CTRadon  # noqa: E402  (import after sys.path tweak)


def _normalize(image):
    image = image - np.min(image)
    max_val = np.max(image)
    if max_val > 0:
        image = image / max_val
    return image


def test_ctradon_matches_iradon_reference():
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (128, 128), mode="reflect", anti_aliasing=True)
    theta = np.linspace(0, 180, 180, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)

    reconstruction = CTRadon(sinogram.T, angle_range=180, output_size=phantom.shape[0])
    reference = iradon(sinogram, theta=theta, circle=True, filter_name="ramp")

    reconstruction_norm = _normalize(reconstruction)
    reference_norm = _normalize(reference)

    mse = np.mean((reconstruction_norm - reference_norm) ** 2)
    assert mse < 5e-3


def test_ctradon_rejects_invalid_shape():
    invalid = np.ones((32,))
    with pytest.raises(ValueError):
        CTRadon(invalid, angle_range=180)

