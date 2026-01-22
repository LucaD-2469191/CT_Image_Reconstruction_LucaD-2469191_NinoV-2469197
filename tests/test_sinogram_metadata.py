import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_slice import CTRadon, _auto_orient_sinogram, _detect_angle_range_from_sinogram  # noqa: E402


def _make_redundant_sinogram(num_angles=180, num_detectors=64):
    rng = np.random.default_rng(0)
    half = rng.normal(size=(num_angles // 2, num_detectors))
    mirrored = np.flip(half, axis=1)[::-1]
    return np.vstack([half, mirrored])


def _make_full_sinogram(num_angles=360, num_detectors=64):
    rng = np.random.default_rng(1)
    return rng.normal(size=(num_angles, num_detectors))


def test_detect_angle_range_prefers_redundancy():
    sino = _make_redundant_sinogram()
    angle = _detect_angle_range_from_sinogram(sino)
    assert angle == 180.0


def test_detect_angle_range_prefers_full_coverage():
    sino = _make_full_sinogram()
    angle = _detect_angle_range_from_sinogram(sino)
    assert angle == 360.0


def test_auto_orientation_transposes_when_angles_in_columns():
    angles = 180
    detectors = 64
    # Store angles along columns to simulate vertical stacking.
    base = np.linspace(0, 1, detectors)[:, None] + np.linspace(0, 1, angles)[None, :]
    sino, orientation = _auto_orient_sinogram(base, sensor_orientation="auto")
    assert sino.shape == (angles, detectors)
    assert orientation == "angles_rows"


def test_ctradon_supports_hamming_and_hann_filters():
    angles = 120
    detectors = 90
    x = np.linspace(0, np.pi, detectors)
    base = np.sin(x)[None, :]
    sinogram = base + np.linspace(0, 1, angles)[:, None] * 0.01

    for name in ("hamming", "hann"):
        reconstruction = CTRadon(
            sinogram,
            angle_range=180,
            filter_name=name,
            output_size=detectors,
            sensor_orientation="angles_rows",
        )
        assert reconstruction.shape == (detectors, detectors)
        assert np.isfinite(reconstruction).all()


def test_auto_orientation_prefers_mod_180_matches_rows():
    angles = 180
    detectors = 220
    base = np.linspace(0, 1, angles)[:, None] + np.linspace(0, 1, detectors)[None, :]
    # Store angles along rows already; expect no transpose.
    oriented, orientation = _auto_orient_sinogram(base, sensor_orientation="auto")
    assert oriented.shape == (angles, detectors)
    assert orientation == "angles_rows"

