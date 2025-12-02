"""
CT Slice Reconstruction using Direct Fourier Reconstruction (DFR)

This module implements the Direct Fourier Reconstruction method based on the
Fourier Slice Theorem for CT image reconstruction from parallel projection sinograms.

The Fourier Slice Theorem states that the 1D Fourier transform of a parallel 
projection of an image (a row in a sinogram) is equal to a slice through the 
2D Fourier transform of the image, passing through the origin at the same angle.

Reference: CT Image Reconstruction Project - Direct Fourier Reconstruction
"""

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import map_coordinates


def _validate_inputs(sinogram, angle_range):
    """
    Normalize and validate sinogram inputs shared by reconstruction algorithms.
    
    Returns
    -------
    sinogram : np.ndarray
        Validated sinogram as float64.
    angle_range : float
        Normalized angular range in degrees.
    num_angles : int
        Number of projection angles.
    num_detectors : int
        Number of detector bins per projection.
    """
    if not isinstance(sinogram, np.ndarray):
        try:
            sinogram = np.array(sinogram)
        except Exception as exc:
            raise TypeError(
                f"Sinogram must be a numpy array or convertible to one. Error: {exc}"
            ) from exc

    if sinogram.ndim != 2:
        raise ValueError(f"Sinogram must be 2D array, got shape {sinogram.shape}")

    num_angles, num_detectors = sinogram.shape

    if num_angles < 2:
        raise ValueError(f"Sinogram must have at least 2 angles, got {num_angles}")

    if num_detectors < 2:
        raise ValueError(f"Sinogram must have at least 2 detectors, got {num_detectors}")

    try:
        angle_range = float(angle_range)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"angle_range must be numeric, got {type(angle_range)}") from exc

    if not (1 <= angle_range <= 360):
        raise ValueError(f"angle_range must be between 1 and 360 degrees, got {angle_range}")

    if not np.issubdtype(sinogram.dtype, np.floating):
        sinogram = sinogram.astype(np.float64)
    else:
        sinogram = sinogram.astype(np.float64, copy=False)

    return sinogram, angle_range, num_angles, num_detectors


def CTSlice(sinogram, angle_range=180):
    """
    Reconstruct a CT slice from a sinogram using Direct Fourier Reconstruction.
    
    This function implements the Fourier Slice Theorem to reconstruct a 2D image
    from its parallel projection sinogram. The algorithm works by:
    1. Taking the 1D FFT of each projection (row) in the sinogram
    2. Placing these frequency slices in a 2D Fourier space at their corresponding angles
    3. Interpolating to fill a Cartesian grid in Fourier space
    4. Taking the 2D inverse FFT to obtain the reconstructed image
    
    This implementation uses scipy's map_coordinates for high-quality interpolation
    from polar to Cartesian coordinates in Fourier space.
    
    Parameters
    ----------
    sinogram : numpy.ndarray
        2D array where each row represents a parallel projection at a specific angle.
        Shape: (num_angles, num_detectors)
        Can be integer type (will be converted to float) or already float.
    angle_range : int or float, optional
        The angular range covered by the sinogram in degrees.
        Typically 180 or 360. Default is 180.
        Must be between 1 and 360.
        
    Returns
    -------
    reconstruction : numpy.ndarray
        The reconstructed 2D CT slice image as float64.
        Shape: (N, N) where N equals num_detectors
        
    Raises
    ------
    ValueError
        If sinogram is not 2D, if angle_range is invalid, or if sinogram is too small
    TypeError
        If sinogram is not a numpy array or cannot be converted to one
        
    Notes
    -----
    - This implementation uses fftshift/ifftshift to maintain correct phase information
    - Works only for parallel projection sinograms
    - The reconstruction size is determined by the sinogram dimensions
    - Uses Hermitian symmetry for 180° sinograms
    - Input sinogram will be converted to float64 if not already
    
    Examples
    --------
    >>> sino = np.load('sinogram.npy')
    >>> reconstructed = CTSlice(sino, angle_range=180)
    >>> 
    >>> # With image file
    >>> import cv2
    >>> sino = cv2.imread('sinogram.png', cv2.IMREAD_GRAYSCALE)
    >>> reconstructed = CTSlice(sino, angle_range=360)
    """
    
    sinogram, angle_range, _, _ = _validate_inputs(sinogram, angle_range)

    # Use the gridded implementation which provides better results
    return CTSlice_gridded(sinogram, angle_range)


def CTSlice_gridded(sinogram, angle_range=180):
    """
    Alternative implementation using gridded interpolation approach.
    
    This version uses scipy's map_coordinates for more accurate interpolation
    when mapping from polar to Cartesian coordinates in Fourier space.
    This is the preferred implementation for production use.
    
    Parameters
    ----------
    sinogram : numpy.ndarray
        2D array where each row represents a parallel projection at a specific angle.
    angle_range : int, optional
        The angular range covered by the sinogram in degrees. Default is 180.
        
    Returns
    -------
    reconstruction : numpy.ndarray
        The reconstructed 2D CT slice image.
    """
    
    num_angles, num_detectors = sinogram.shape
    angles = np.linspace(0, np.deg2rad(angle_range), num_angles, endpoint=False)
    
    # Output size - make it same as detector count
    output_size = num_detectors
    
    # Compute 1D FFT for all projections
    projections_fft = np.zeros((num_angles, num_detectors), dtype=complex)
    for i in range(num_angles):
        projections_fft[i] = fftshift(fft(ifftshift(sinogram[i])))
    
    # Create 2D Cartesian frequency grid centered at origin
    center = output_size // 2
    kx = np.arange(output_size) - center
    ky = np.arange(output_size) - center
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # Convert Cartesian coordinates to polar
    r_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    theta_grid = np.arctan2(ky_grid, kx_grid)
    
    # Handle angle wrapping for proper interpolation
    # For 180 degree range, we use symmetry: F(-θ) = F*(θ)
    # For 360 degree range, we cover the full circle
    if angle_range == 180:
        # Use Hermitian symmetry property
        theta_grid = np.mod(theta_grid, np.pi)
    else:
        theta_grid = np.mod(theta_grid, 2*np.pi)
    
    # Map to indices in our projections_fft array
    # Angle index: map angle to projection index
    angle_indices = theta_grid * num_angles / np.deg2rad(angle_range)
    angle_indices = np.clip(angle_indices, 0, num_angles - 1)
    
    # Radial index: map radial distance to frequency index
    # The center of projections_fft corresponds to DC (zero frequency)
    # which is at index num_detectors // 2 after fftshift
    max_radius = num_detectors // 2
    freq_indices = r_grid + num_detectors // 2
    freq_indices = np.clip(freq_indices, 0, num_detectors - 1)
    
    # Use map_coordinates for bilinear interpolation
    # This smoothly interpolates between the polar-sampled frequency data
    coordinates = np.array([angle_indices.ravel(), freq_indices.ravel()])
    fourier_2d_real = map_coordinates(projections_fft.real, coordinates, order=1, mode='nearest')
    fourier_2d_imag = map_coordinates(projections_fft.imag, coordinates, order=1, mode='nearest')
    fourier_2d = (fourier_2d_real + 1j * fourier_2d_imag).reshape((output_size, output_size))
    
    # Perform 2D inverse FFT to get reconstructed image
    reconstruction = fftshift(ifft2(ifftshift(fourier_2d)))
    reconstruction = np.real(reconstruction)
    
    return reconstruction


def _get_fourier_filter(size, filter_name="ram-lak"):
    """
    Construct frequency-domain filters for filtered backprojection.
    """
    if size % 2 == 1:
        raise ValueError("Filter size must be even.")

    valid_filters = {"ram-lak", "ramp", "shepp-logan", "cosine", "hamming", "hann", None, "none"}
    if filter_name not in valid_filters:
        raise ValueError(f"Unknown filter '{filter_name}'. Valid options: {valid_filters}")

    n = np.concatenate(
        (
            np.arange(1, size // 2 + 1, 2, dtype=int),
            np.arange(size // 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(fft(f))

    if filter_name in {"ram-lak", "ramp"}:
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        fourier_filter *= fftshift(np.sin(freq))
    elif filter_name == "hamming":
        fourier_filter *= fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftshift(np.hanning(size))
    elif filter_name in {None, "none"}:
        fourier_filter[:] = 1

    return fourier_filter.astype(np.float64)


def _filter_sinogram(sinogram, filter_name="ram-lak"):
    """
    Apply frequency-domain filtering to each projection in the sinogram.
    """
    _, num_detectors = sinogram.shape
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * num_detectors))))
    pad_width = ((0, 0), (0, padded_size - num_detectors))
    sinogram_padded = np.pad(sinogram, pad_width, mode="constant", constant_values=0)

    fourier_filter = _get_fourier_filter(padded_size, filter_name)
    projections = fft(sinogram_padded, axis=1) * fourier_filter
    filtered = np.real(ifft(projections, axis=1))[:, :num_detectors]

    return filtered


def _sinogram_circle_to_square(sinogram):
    """
    Pad sinogram detectors dimension so that the inscribed circle maps to a square grid.
    """
    num_detectors = sinogram.shape[1]
    diagonal = int(np.ceil(np.sqrt(2) * num_detectors))
    pad = diagonal - num_detectors
    if pad <= 0:
        return sinogram

    old_center = num_detectors // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((0, 0), (pad_before, pad - pad_before))
    return np.pad(sinogram, pad_width, mode="constant", constant_values=0)


def _backproject(filtered, theta_radians, output_size, crop_to_circle=True):
    """
    Perform the backprojection integral using linear interpolation.
    """
    reconstruction = np.zeros((output_size, output_size), dtype=np.float64)
    radius = output_size // 2
    x_coords, y_coords = np.mgrid[:output_size, :output_size] - radius

    detector_positions = np.arange(filtered.shape[1], dtype=np.float64) - filtered.shape[1] // 2

    for projection, angle in zip(filtered, theta_radians):
        coord = y_coords * np.cos(angle) - x_coords * np.sin(angle)
        interpolated = np.interp(coord.ravel(), detector_positions, projection, left=0.0, right=0.0)
        reconstruction += interpolated.reshape(reconstruction.shape)

    if crop_to_circle:
        mask = (x_coords**2 + y_coords**2) > radius**2
        reconstruction[mask] = 0.0

    return reconstruction


def CTRadon(
    sinogram,
    angle_range=180,
    filter_name="ram-lak",
    output_size=None,
    crop_to_circle=True,
):
    """
    Reconstruct a CT slice using Filtered Backprojection (FBP).

    Parameters
    ----------
    sinogram : numpy.ndarray
        2D array where each row represents a parallel projection at a specific angle.
        Shape: (num_angles, num_detectors)
    angle_range : int or float, optional
        Total angular coverage in degrees. Default is 180. Must be in (0, 360].
    filter_name : str or None, optional
        Filter to apply in Fourier domain. Supported: 'ram-lak', 'ramp',
        'shepp-logan', 'cosine', 'hamming', 'hann', or None for no filtering.
    output_size : int, optional
        Number of rows/columns in the reconstructed slice. Defaults to number of detectors.
    crop_to_circle : bool, optional
        If True (default) zeroes pixels outside the inscribed circle.

    Returns
    -------
    numpy.ndarray
        Reconstructed CT slice using the FBP method.
    """

    sinogram, angle_range, num_angles, num_detectors = _validate_inputs(sinogram, angle_range)

    working_sinogram = sinogram
    detector_extent = num_detectors
    if crop_to_circle:
        working_sinogram = _sinogram_circle_to_square(sinogram)
        detector_extent = working_sinogram.shape[1]

    if output_size is None:
        if crop_to_circle:
            output_size = detector_extent
        else:
            output_size = int(np.floor(np.sqrt(detector_extent**2 / 2.0)))
            output_size = max(output_size, 1)

    filtered = _filter_sinogram(working_sinogram, filter_name=filter_name)
    theta_degrees = np.linspace(0.0, angle_range, num_angles, endpoint=False)
    theta_radians = np.deg2rad(theta_degrees)
    reconstruction = _backproject(filtered, theta_radians, output_size=output_size, crop_to_circle=crop_to_circle)

    # Scale by angular step size (matches analytical inverse Radon transform)
    reconstruction *= np.deg2rad(angle_range) / (2 * num_angles)

    return reconstruction


if __name__ == "__main__":
    # Simple test with a synthetic sinogram
    print("Testing CTSlice implementation...")
    
    # Create a simple test case: sinogram of a point (should reconstruct to a point)
    test_size = 128
    test_angles = 180
    
    # Create a simple synthetic sinogram (e.g., all zeros with a peak in the middle)
    test_sinogram = np.zeros((test_angles, test_size))
    test_sinogram[:, test_size//2] = 1.0  # Peak in the center
    
    # Test reconstruction
    result = CTSlice(test_sinogram, angle_range=180)
    
    print(f"Input sinogram shape: {test_sinogram.shape}")
    print(f"Output reconstruction shape: {result.shape}")
    print(f"Reconstruction value range: [{result.min():.4f}, {result.max():.4f}]")
    print("\nTest completed successfully!")

