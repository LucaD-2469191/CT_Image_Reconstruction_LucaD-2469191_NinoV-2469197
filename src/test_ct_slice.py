"""
Test script for CTSlice Direct Fourier Reconstruction

This script tests the CTSlice implementation with various sinograms
from the parallel projection dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from ct_slice import CTSlice

# Try to import skimage's iradon for comparison
try:
    from skimage.transform import iradon
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not available. Install with: pip install scikit-image")


def load_sinogram_image(filepath):
    """Load a sinogram from an image file."""
    # Load as grayscale
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    
    # Convert to float and normalize
    sinogram = img.astype(np.float64) / 255.0
    return sinogram


def test_sinogram(sinogram_path, angle_range=180, save_results=True):
    """
    Test CTSlice reconstruction on a single sinogram.
    
    Parameters
    ----------
    sinogram_path : Path or str
        Path to the sinogram image file
    angle_range : int
        Angular range (180 or 360 degrees)
    save_results : bool
        Whether to save the reconstruction results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {Path(sinogram_path).name}")
    print(f"{'='*60}")
    
    # Load sinogram
    sinogram = load_sinogram_image(sinogram_path)
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Sinogram value range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")
    
    # Test CTSlice implementation
    print("\nTesting CTSlice (Direct Fourier Reconstruction)...")
    reconstruction_dfr = CTSlice(sinogram, angle_range=angle_range)
    print(f"Reconstruction shape: {reconstruction_dfr.shape}")
    print(f"Reconstruction value range: [{reconstruction_dfr.min():.4f}, {reconstruction_dfr.max():.4f}]")
    
    # Compare with skimage's iradon if available
    reconstruction_skimage = None
    if HAS_SKIMAGE:
        print("\nTesting skimage's iradon (reference implementation)...")
        # skimage expects sinogram transposed and angles in degrees
        theta = np.linspace(0, angle_range, sinogram.shape[0], endpoint=False)
        reconstruction_skimage = iradon(sinogram.T, theta=theta, circle=True)
        print(f"Reconstruction shape: {reconstruction_skimage.shape}")
        print(f"Reconstruction value range: [{reconstruction_skimage.min():.4f}, {reconstruction_skimage.max():.4f}]")
    else:
        print("\nSkipping skimage comparison (not installed)")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Sinogram
    axes[0, 0].imshow(sinogram, cmap='gray', aspect='auto')
    axes[0, 0].set_title(f'Sinogram: {Path(sinogram_path).name}', fontsize=10)
    axes[0, 0].set_xlabel('Detector', fontsize=8)
    axes[0, 0].set_ylabel('Angle', fontsize=8)
    axes[0, 0].tick_params(labelsize=7)
    
    # Our CTSlice (DFR) reconstruction
    axes[0, 1].imshow(reconstruction_dfr, cmap='gray')
    axes[0, 1].set_title('CTSlice (Direct Fourier)', fontsize=10)
    axes[0, 1].axis('off')
    
    # Skimage reconstruction or placeholder
    if reconstruction_skimage is not None:
        axes[1, 0].imshow(reconstruction_skimage, cmap='gray')
        axes[1, 0].set_title('skimage iradon (FBP)', fontsize=10)
        axes[1, 0].axis('off')
        
        # Difference between methods (normalize both for fair comparison)
        dfr_norm = reconstruction_dfr / (np.abs(reconstruction_dfr).max() + 1e-10)
        ski_norm = reconstruction_skimage / (np.abs(reconstruction_skimage).max() + 1e-10)
        diff = np.abs(dfr_norm - ski_norm)
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title(f'Normalized Difference\n(Max: {diff.max():.4f})', fontsize=10)
        axes[1, 1].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'skimage not installed\nInstall with:\npip install scikit-image', 
                       ha='center', va='center', fontsize=9)
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Comparison not available', 
                       ha='center', va='center', fontsize=9)
        axes[1, 1].axis('off')
    
    plt.tight_layout(pad=0.5)
    
    if save_results:
        # Save figure with lower DPI
        output_path = Path('../results') / f'test_{Path(sinogram_path).stem}_reconstruction.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\nSaved results to: {output_path}")
        
        # Save reconstruction as image
        # Normalize to 0-255 range for saving
        recon_normalized = reconstruction_dfr - reconstruction_dfr.min()
        if recon_normalized.max() > 0:
            recon_normalized = recon_normalized / recon_normalized.max() * 255
        recon_normalized = recon_normalized.astype(np.uint8)
        
        recon_path = Path('../results') / f'{Path(sinogram_path).stem}_reconstructed.png'
        cv2.imwrite(str(recon_path), recon_normalized)
        print(f"Saved reconstruction to: {recon_path}")
    
    plt.show()
    
    return reconstruction_dfr, reconstruction_skimage


def main():
    """Run tests on available sinograms."""
    
    # Define data directory
    data_dir = Path('../Data/Parallel Projection')
    
    # List of sinogram files to test
    # Based on the file names, these appear to be sinograms
    sinogram_files = [
        'sino_42.png',
        'sino_circle.png', 
        'sino_drawing.png',
        'sino.jpg',
    ]
    
    # Also test the other images which might be sinograms or reference images
    other_files = [
        'SheppLoganPhantom.png',
        'Lotus.png',
        'Walnut.png',
    ]
    
    print("CT Slice - Direct Fourier Reconstruction Test Suite")
    print("="*60)
    
    # Test sinogram files (assuming 180 degree range)
    for filename in sinogram_files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                test_sinogram(filepath, angle_range=180, save_results=True)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()

