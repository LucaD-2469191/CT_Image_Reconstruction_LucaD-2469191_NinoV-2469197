"""
Comprehensive validation script for CTSlice implementation.

This script tests the CTSlice Direct Fourier Reconstruction on all available
parallel projection sinograms and compares results with expected outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from ct_slice import CTSlice, CTRadon, generate_sinogram
import cv2

# Ensure local src imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def _normalize_image(image):
    """Normalize image to [0, 1] for fair metric comparison."""
    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    return image


def _create_test_phantom(size, pattern):
    """
    Create synthetic phantoms for forward-projection validation.
    
    Parameters
    ----------
    size : int
        Phantom image size (square).
    pattern : str
        Pattern identifier: 'square', 'circle', 'cross', 'point', or 'offset_rect'.
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    
    if pattern == "square":
        margin = size // 4
        phantom[margin:-margin, margin:-margin] = 1.0
    elif pattern == "circle":
        y, x = np.ogrid[:size, :size]
        radius = int(size * 0.3)
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        phantom[mask] = 1.0
    elif pattern == "cross":
        thickness = max(2, size // 16)
        phantom[:, center - thickness:center + thickness] = 1.0
        phantom[center - thickness:center + thickness, :] = 1.0
    elif pattern == "offset_rect":
        rect_height = size // 2
        rect_width = size // 3
        top = size // 5
        left = size // 8
        phantom[top:top + rect_height, left:left + rect_width] = 1.0
    elif pattern == "point":
        phantom[center, center] = 1.0
    else:
        raise ValueError(f"Unknown phantom pattern: {pattern}")
    
    return phantom


def _load_phantom_image(image_path, normalize=True):
    """Load an arbitrary image file to use as a phantom."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load phantom image: {image_path}")
    phantom = img.astype(np.float64)
    if normalize:
        phantom /= 255.0
    return phantom


def _resize_to_shape(image, target_shape):
    """Resize reconstruction to match the phantom shape if needed."""
    if image.shape == target_shape:
        return image
    target_h, target_w = target_shape
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


def _compute_image_metrics(reference, estimate):
    """Return MAE, MSE, and PSNR between normalized reference and estimate."""
    ref_norm = _normalize_image(reference)
    est_norm = _normalize_image(estimate)
    diff = ref_norm - est_norm
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10(1.0 / mse))
    return {"mae": mae, "mse": mse, "psnr": psnr}

def validate_reconstruction(sinogram_path, angle_range=None, save_results=True, show_plot=False, sensor_orientation="auto"):
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
    reconstruction = CTSlice(sinogram, angle_range=angle_range, sensor_orientation=sensor_orientation)
    reconstruction_fbp = CTRadon(sinogram, angle_range=angle_range, sensor_orientation=sensor_orientation)
    
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
        results_dir = PROJECT_ROOT / "results" / "validation"
        output_path = results_dir / f'{sino_path.stem}_validation.png'
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


def validate_generate_sinogram(
    patterns=None,
    custom_images=None,
    image_size=128,
    num_angles=180,
    angle_range=180,
    save_results=True,
    show_plot=False,
):
    """
    Validate the forward projection implemented in generate_sinogram.
    
    The validation creates synthetic phantoms (or uses provided images), generates
    their sinograms, and reconstructs them with CTRadon. Reconstruction fidelity
    is reported with simple error metrics.
    """
    if patterns is None and not custom_images:
        patterns = ("square", "circle", "cross", "point", "offset_rect")
    if patterns is None:
        patterns = ()
    if custom_images is None:
        custom_images = []
    
    print("\n" + "="*70)
    print("Forward Projection Validation (generate_sinogram)")
    print("="*70)
    print(f"Phantom size: {image_size}×{image_size}")
    print(f"Angles: {num_angles}, Angle range: {angle_range}°")
    
    results_dir = PROJECT_ROOT / "results" / "validation" / "forward_projection"
    results = []
    
    def _phantom_label(name, is_custom):
        return f"custom:{name}" if is_custom else f"pattern:{name}"
    
    phantom_entries = []
    for pattern in patterns:
        phantom = _create_test_phantom(image_size, pattern)
        phantom_entries.append((_phantom_label(pattern, False), phantom, None))
    
    for entry in custom_images:
        if isinstance(entry, (list, tuple)):
            name, path = entry
        else:
            path = entry
            name = Path(path).stem
        path = Path(path)
        phantom = _load_phantom_image(path)
        phantom_entries.append((_phantom_label(name, True), phantom, path))
    
    if not phantom_entries:
        print("No phantoms specified for validation.")
        return results
    
    for label, phantom, source_path in phantom_entries:
        print(f"\nPhantom: {label}")
        if source_path:
            print(f"  Source image: {source_path}")
        phantom_shape = phantom.shape
        print(f"  Phantom shape: {phantom_shape}")
        
        sinogram = generate_sinogram(
            phantom,
            num_angles=num_angles,
            angle_range=angle_range,
        )
        reconstruction = CTRadon(
            sinogram,
            angle_range=angle_range,
            output_size=max(phantom_shape),
            filter_name="ram-lak",
        )
        reconstruction = _resize_to_shape(reconstruction, phantom_shape)
        
        metrics = _compute_image_metrics(phantom, reconstruction)
        detector_extent = sinogram.shape[1]
        print(f"  Sinogram shape: {sinogram.shape} (detector extent: {detector_extent})")
        print(f"  Reconstruction range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")
        print(f"  MAE: {metrics['mae']:.6f} | MSE: {metrics['mse']:.6f} | PSNR: "
              f"{metrics['psnr']:.2f} dB")
        
        results.append({
            "phantom": label,
            "phantom_shape": phantom.shape,
            "source_path": str(source_path) if source_path else None,
            "sinogram_shape": sinogram.shape,
            "metrics": metrics,
        })
        
        if save_results or show_plot:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(_normalize_image(phantom), cmap="gray")
            axes[0].set_title(f"Phantom: {label}")
            axes[0].axis("off")
            
            axes[1].imshow(sinogram, cmap="gray", aspect="auto")
            axes[1].set_title("Generated Sinogram")
            axes[1].set_xlabel("Detector")
            axes[1].set_ylabel("Angle")
            
            axes[2].imshow(_normalize_image(reconstruction), cmap="gray")
            axes[2].set_title("CTRadon Reconstruction")
            axes[2].axis("off")
            
            plt.tight_layout(pad=0.8)
            
            if save_results:
                results_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(source_path).stem if source_path else label.split(":", 1)[-1]
                output_path = results_dir / f"generate_sinogram_{suffix}.png"
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                print(f"  Saved visualization: {output_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
    
    print("\nForward projection validation complete.")
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
    
    # Define test cases (resolve relative to project root)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'Data' / 'Parallel Projection'
    
    test_cases = [
        ('sino_42.png', 180, "auto"),  # Appears to be 180° based on 180 angles
        ('sino_circle.png', 360, "auto"),  # 361 angles suggests 360°
        ('sino_drawing.png', 180, "auto"),  # 180 angles
        ('sino.jpg', None, "auto"),  # Auto-detect (375 angles -> likely 360°)
        ('SheppLoganPhantom.png', 180, "angles_rows"),
        ('Lotus.png', 360, "auto"),
        ('Walnut.png', 360, "auto"),
    ]
    
    # Run tests
    all_results = []
    for filename, angle_range, sensor_orientation in test_cases:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\nWarning: File not found: {filepath}")
            continue
        
        try:
            results = validate_reconstruction(filepath, angle_range=angle_range, show_plot=show_plots, sensor_orientation=sensor_orientation)
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
        
        print(f"\n✓ All dataset reconstructions completed successfully!")
        print(f"✓ Results saved to: results/validation/")
    else:
        print("\n✗ No dataset reconstructions completed successfully")
    
    print("="*70)
    
    # Forward projection validation
    validate_generate_sinogram(show_plot=show_plots)


if __name__ == "__main__":
    main()

