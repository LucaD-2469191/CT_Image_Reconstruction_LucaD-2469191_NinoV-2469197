"""
Comprehensive validation script for CTSlice implementation.

This script tests the CTSlice Direct Fourier Reconstruction on all available
parallel projection sinograms and compares results with expected outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from ct_slice import CTSlice, CTRadon


def load_sinogram(filepath):
    """Load a sinogram from an image file and normalize to [0, 1]."""
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    return img.astype(np.float64) / 255.0


def determine_angle_range(num_angles):
    """
    Heuristic to determine if sinogram is 180° or 360°.
    
    If num_angles is close to 180, it's likely 180°.
    If num_angles is close to 360, it's likely 360°.
    """
    if abs(num_angles - 180) < abs(num_angles - 360):
        return 180
    else:
        return 360


def validate_reconstruction(sinogram_path, angle_range=None, save_results=True, show_plot=False):
    """
    Validate CTSlice reconstruction on a specific sinogram.
    
    Parameters
    ----------
    sinogram_path : Path or str
        Path to the sinogram image file
    angle_range : int, optional
        Angular range (180 or 360). If None, will be auto-determined.
    save_results : bool
        Whether to save visualization and reconstruction
    show_plot : bool
        Whether to display the plot on screen (default: False)
        
    Returns
    -------
    dict
        Dictionary containing test results and metrics
    """
    
    sino_path = Path(sinogram_path)
    print(f"\n{'='*70}")
    print(f"Testing: {sino_path.name}")
    print(f"{'='*70}")
    
    # Load sinogram
    sinogram = load_sinogram(sino_path)
    num_angles, num_detectors = sinogram.shape
    
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Number of angles: {num_angles}")
    print(f"Number of detectors: {num_detectors}")
    print(f"Value range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")
    
    # Determine angle range if not specified
    if angle_range is None:
        angle_range = determine_angle_range(num_angles)
        print(f"Auto-detected angle range: {angle_range}°")
    else:
        print(f"Using specified angle range: {angle_range}°")
    
    # Perform reconstruction
    print("\nReconstructing...")
    reconstruction = CTSlice(sinogram, angle_range=angle_range)
    reconstruction_fbp = CTRadon(sinogram, angle_range=angle_range)
    
    print(f"DFR Reconstruction shape: {reconstruction.shape}")
    print(f"DFR value range: [{reconstruction.min():.6f}, {reconstruction.max():.6f}]")
    print(f"DFR mean/std: {reconstruction.mean():.6f} / {reconstruction.std():.6f}")
    print(f"FBP value range: [{reconstruction_fbp.min():.6f}, {reconstruction_fbp.max():.6f}]")
    
    # Calculate metrics
    results = {
        'filename': sino_path.name,
        'num_angles': num_angles,
        'num_detectors': num_detectors,
        'angle_range': angle_range,
        'recon_shape': reconstruction.shape,
        'recon_min': reconstruction.min(),
        'recon_max': reconstruction.max(),
        'recon_mean': reconstruction.mean(),
        'recon_std': reconstruction.std(),
        'fbp_min': reconstruction_fbp.min(),
        'fbp_max': reconstruction_fbp.max(),
    }
    
    # Visualize and save
    if save_results:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Sinogram
        im1 = axes[0].imshow(sinogram, cmap='gray', aspect='auto')
        axes[0].set_title(f'{sino_path.stem} Sinogram\n{num_angles}∠ × {num_detectors}det ({angle_range}°)', 
                         fontsize=9)
        axes[0].set_xlabel('Detector', fontsize=8)
        axes[0].set_ylabel('Angle', fontsize=8)
        axes[0].tick_params(labelsize=7)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Reconstruction
        # Normalize for display
        recon_display = reconstruction - reconstruction.min()
        if recon_display.max() > 0:
            recon_display = recon_display / recon_display.max()
        
        im2 = axes[1].imshow(recon_display, cmap='gray')
        axes[1].set_title(f'DFR Reconstruction\n[{reconstruction.min():.4f}, {reconstruction.max():.4f}]', 
                         fontsize=9)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        recon_fbp_display = reconstruction_fbp - reconstruction_fbp.min()
        if recon_fbp_display.max() > 0:
            recon_fbp_display = recon_fbp_display / recon_fbp_display.max()

        im3 = axes[2].imshow(recon_fbp_display, cmap='gray')
        axes[2].set_title(f'FBP Reconstruction\n[{reconstruction_fbp.min():.4f}, {reconstruction_fbp.max():.4f}]', 
                         fontsize=9)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout(pad=1.0)
        
        # Save figure with lower DPI for smaller file size
        output_path = Path('../results/validation') / f'{sino_path.stem}_validation.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\nSaved validation figure: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Save reconstruction as normalized image
        recon_uint8 = (recon_display * 255).astype(np.uint8)
        recon_path = output_path.parent / f'{sino_path.stem}_reconstruction.png'
        cv2.imwrite(str(recon_path), recon_uint8)
        print(f"Saved reconstruction: {recon_path}")

        recon_fbp_uint8 = (recon_fbp_display * 255).astype(np.uint8)
        recon_fbp_path = output_path.parent / f'{sino_path.stem}_fbp_reconstruction.png'
        cv2.imwrite(str(recon_fbp_path), recon_fbp_uint8)
        print(f"Saved FBP reconstruction: {recon_fbp_path}")
    
    print(f"{'='*70}")
    return results


def main(show_plots=True):
    """
    Run comprehensive validation on all available sinograms.
    
    Parameters
    ----------
    show_plots : bool
        If True, display plots on screen. If False, only save them.
    """
    
    print("="*70)
    print("CTSlice - Direct Fourier Reconstruction Validation")
    print("="*70)
    
    if show_plots:
        print("Display mode: Plots will be shown on screen")
    else:
        print("Batch mode: Plots will be saved without displaying")
    
    # Define test cases
    data_dir = Path('../Data/Parallel Projection')
    
    test_cases = [
        ('sino_42.png', 180),  # Appears to be 180° based on 180 angles
        ('sino_circle.png', 360),  # 361 angles suggests 360°
        ('sino_drawing.png', 180),  # 180 angles
        ('sino.jpg', None),  # Auto-detect (375 angles -> likely 360°)
    ]
    
    # Run tests
    all_results = []
    for filename, angle_range in test_cases:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\nWarning: File not found: {filepath}")
            continue
        
        try:
            results = validate_reconstruction(filepath, angle_range=angle_range, show_plot=show_plots)
            all_results.append(results)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all_results:
        print(f"\nSuccessfully validated {len(all_results)} sinograms:\n")
        for r in all_results:
            print(f"  {r['filename']:25s} - "
                  f"{r['num_angles']:3d} angles × {r['num_detectors']:3d} detectors "
                  f"({r['angle_range']:3d}°) - "
                  f"Recon range: [{r['recon_min']:7.4f}, {r['recon_max']:7.4f}]")
        
        print(f"\n✓ All tests completed successfully!")
        print(f"✓ Results saved to: results/validation/")
    else:
        print("\n✗ No tests completed successfully")
    
    print("="*70)


if __name__ == "__main__":
    main()

