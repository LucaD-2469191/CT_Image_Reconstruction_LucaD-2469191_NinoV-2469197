import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_slice import fanbeam_to_parallel  # noqa: E402


def _analytic_parallel(theta, s):
    return np.sin(theta) + 0.05 * s


def test_fanbeam_to_parallel_recovers_analytic_function():
    fod = 750.0
    fdd = 1200.0
    sensor_width = 1.25
    center_offset = 2.5
    num_angles = 360
    num_detectors = 320

    beta = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False)
    detector_center = (num_detectors - 1) / 2.0
    detector_indices = np.arange(num_detectors, dtype=np.float64)
    detector_positions = (detector_indices - detector_center - center_offset) * sensor_width
    gamma = np.arctan2(detector_positions, fdd)
    s = fod * np.sin(gamma)

    fan = np.empty((num_angles, num_detectors), dtype=np.float64)
    for i, beta_i in enumerate(beta):
        theta = (beta_i + gamma) % (2 * np.pi)
        fan[i] = _analytic_parallel(theta, s)

    rebinned, theta_grid, s_grid = fanbeam_to_parallel(
        fan,
        fod,
        fdd,
        sensor_width,
        center_offset=center_offset,
        angle_range=360.0,
        y_axis_down=False,
        return_grids=True,
    )

    expected = _analytic_parallel(theta_grid[:, None], s_grid[None, :])
    max_abs_err = np.max(np.abs(rebinned - expected))
    assert max_abs_err < 1.5e-2


def _parallel_to_fan(parallel_sino, fod, fdd, sensor_width, angle_range, *, upsample_angles=2):
    num_angles, num_detectors = parallel_sino.shape
    angle_range_rad = np.deg2rad(angle_range)
    theta_step = angle_range_rad / num_angles
    detector_center = (num_detectors - 1) / 2.0

    # Match the physical coverage of the parallel detectors by tying fod/fdd to detector extent.
    max_pos = detector_center * sensor_width
    # Ensure s_limit equals the physical detector extent for the chosen geometry.
    fod = np.sqrt(max_pos**2 + fdd**2)
    fan_angles = int(num_angles * upsample_angles)
    beta = np.linspace(0.0, angle_range_rad, fan_angles, endpoint=False)
    detector_indices = np.arange(num_detectors, dtype=np.float64)
    detector_positions = (detector_indices - detector_center) * sensor_width
    gamma = np.arctan2(detector_positions, fdd)
    s_limit = fod * np.sin(np.max(np.abs(gamma)))
    s_spacing = 2 * s_limit / (num_detectors - 1)

    theta_needed = (beta[:, None] + gamma[None, :]) % angle_range_rad
    s_needed = np.broadcast_to(fod * np.sin(gamma), (fan_angles, num_detectors))

    theta_idx = theta_needed / theta_step
    s_idx = s_needed / s_spacing + detector_center

    periodic_parallel = np.vstack([parallel_sino, parallel_sino[:1]])
    coords = np.array([theta_idx.ravel(), s_idx.ravel()])
    fan = map_coordinates(
        periodic_parallel,
        coords,
        order=1,
        mode="nearest",
    ).reshape(fan_angles, num_detectors)

    return fan, fod


def test_fanbeam_to_parallel_roundtrip_radon_sinogram():
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (64, 64), mode="reflect", anti_aliasing=True)

    theta = np.linspace(0, 180, 180, endpoint=False)
    parallel = radon(phantom, theta=theta, circle=True).T

    sensor_width = 1.0
    fdd = 900.0
    fan, fod = _parallel_to_fan(parallel, fod=1.0, fdd=fdd, sensor_width=sensor_width, angle_range=180.0)

    rebinned = fanbeam_to_parallel(
        fan,
        fod,
        fdd,
        sensor_width,
        angle_range=180.0,
        y_axis_down=False,
        output_num_angles=parallel.shape[0],
        output_num_detectors=parallel.shape[1],
    )

    # Normalize before comparison to avoid scale ambiguity.
    def _normalize(arr):
        arr = arr - arr.min()
        max_val = arr.max()
        return arr / max_val if max_val > 0 else arr

    rebinned_norm = _normalize(rebinned)
    parallel_norm = _normalize(parallel)
    mse = np.mean((rebinned_norm - parallel_norm) ** 2)
    assert mse < 5e-3


def test_fanbeam_to_parallel_supports_angle_upsampling():
    fod = 1e6
    fdd = 1e6
    sensor_width = 1.0
    num_angles = 45
    num_detectors = 32

    beta = np.linspace(0.0, np.deg2rad(180.0), num_angles, endpoint=False)
    base_profile = np.sin(beta)
    fan = base_profile[:, None] * np.ones((1, num_detectors), dtype=np.float64)

    upsampled = fanbeam_to_parallel(
        fan,
        fod,
        fdd,
        sensor_width,
        angle_range=180.0,
        output_num_angles=num_angles * 4,
        y_axis_down=False,
    )

    target_angles = np.linspace(0.0, np.deg2rad(180.0), num_angles * 4, endpoint=False)
    input_angles = np.linspace(0.0, np.deg2rad(180.0), num_angles, endpoint=False)
    # Extend for wrap-around interpolation.
    input_angles_ext = np.concatenate([input_angles, [np.deg2rad(180.0)]])
    base_ext = np.concatenate([base_profile, [base_profile[0]]])
    expected = np.interp(target_angles, input_angles_ext, base_ext)

    max_abs_err = np.max(np.abs(upsampled[:, 0] - expected))
    assert max_abs_err < 1e-2

