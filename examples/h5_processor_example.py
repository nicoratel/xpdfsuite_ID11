"""
Example: Using the H5 Processor programmatically (without Streamlit)

This example demonstrates how to use the core H5 processing functions
directly in Python code without the Streamlit interface.
"""

import numpy as np
from xpdfsuite_ID11 import XRDProcessor
from xpdfsuite_ID11.tabs_h5.extract_pdf_h5 import calculate_pdf


def example_basic_integration():
    """Example 1: Basic I(q) integration from H5 file."""
    
    # Configuration
    sample_h5 = "/path/to/sample.h5"
    poni_file = "/path/to/calibration.poni"
    
    # Create processor instance
    processor = XRDProcessor(
        h5_file=sample_h5,
        frame=0,  # First frame, or "mean" for average
        poni_file=poni_file,
        polarization_factor=0.95,
        entry=1,  # HDF5 entry number
    )
    
    # Integrate to get I(q)
    q, I_q = processor.integrate()
    
    print(f"Q range: {q.min():.3f} - {q.max():.3f} Å⁻¹")
    print(f"I(q) max: {I_q.max():.4e}")
    
    return q, I_q


def example_with_corrections():
    """Example 2: Integration with dark and flat corrections."""
    
    sample_h5 = "/path/to/sample.h5"
    ref_h5 = "/path/to/reference.h5"
    poni_file = "/path/to/calibration.poni"
    dark_file = "/path/to/dark.edf"
    flat_file = "/path/to/flat.edf"
    mask_file = "/path/to/mask.edf"
    
    # Process sample with corrections
    processor_sample = XRDProcessor(
        h5_file=sample_h5,
        frame=5,
        poni_file=poni_file,
        dark=dark_file,
        flat=flat_file,
        mask=mask_file,
        polarization_factor=0.95,
        entry=1,
    )
    q_sample, I_sample = processor_sample.integrate()
    
    # Process reference
    processor_ref = XRDProcessor(
        h5_file=ref_h5,
        frame="mean",  # Average all frames
        poni_file=poni_file,
        dark=dark_file,
        flat=flat_file,
        mask=mask_file,
        polarization_factor=0.95,
        entry=1,
    )
    q_ref, I_ref = processor_ref.integrate()
    
    # Interpolate reference to sample q grid
    I_ref_interp = np.interp(q_sample, q_ref, I_ref)
    
    # Background subtraction with scale factor
    bgscale = 1.1
    I_corrected = I_sample - bgscale * I_ref_interp
    
    print(f"Sample I(q) max: {I_sample.max():.4e}")
    print(f"Reference I(q) max: {I_ref.max():.4e}")
    print(f"Corrected I(q) max: {I_corrected.max():.4e}")
    
    return q_sample, I_corrected


def example_pdf_extraction():
    """Example 3: Extract PDF from I(q) data."""
    
    # Assuming we have q and I_q from previous example
    q = np.linspace(0.5, 20, 500)
    I_q = np.exp(-0.1 * q) * np.sin(2 * np.pi * q)  # Dummy data
    
    # PDF extraction parameters
    r_min = 0.1
    r_max = 50.0
    n_points = 500
    Q_damp = 0.03
    window = "hann"
    
    # Calculate PDF
    r, G_r = calculate_pdf(
        q=q,
        I_q=I_q,
        r_min=r_min,
        r_max=r_max,
        n_points=n_points,
        Q_damp=Q_damp,
        window=window,
    )
    
    # Find first peak
    first_peak_idx = np.argmax(G_r)
    first_peak_r = r[first_peak_idx]
    first_peak_G = G_r[first_peak_idx]
    
    print(f"First peak position: {first_peak_r:.3f} Å")
    print(f"First peak height: {first_peak_G:.4e}")
    print(f"r range: {r[0]:.2f} - {r[-1]:.2f} Å")
    
    return r, G_r


def example_full_workflow():
    """Example 4: Complete workflow from H5 to PDF."""
    
    # File paths
    sample_h5 = "/path/to/sample.h5"
    ref_h5 = "/path/to/reference.h5"
    poni_file = "/path/to/calibration.poni"
    dark_file = "/path/to/dark.edf"
    flat_file = "/path/to/flat.edf"
    
    # Step 1: Integrate sample
    print("Step 1: Integrating sample...")
    processor_sample = XRDProcessor(
        sample_h5, frame=0, poni_file=poni_file,
        dark=dark_file, flat=flat_file,
        polarization_factor=0.95, entry=1,
    )
    q, I_sample = processor_sample.integrate()
    
    # Step 2: Integrate reference
    print("Step 2: Integrating reference...")
    processor_ref = XRDProcessor(
        ref_h5, frame="mean", poni_file=poni_file,
        dark=dark_file, flat=flat_file,
        polarization_factor=0.95, entry=1,
    )
    q_ref, I_ref = processor_ref.integrate()
    I_ref = np.interp(q, q_ref, I_ref)
    
    # Step 3: Subtract background
    print("Step 3: Subtracting background...")
    bgscale = 1.0
    I_corrected = I_sample - bgscale * I_ref
    
    # Step 4: Extract PDF
    print("Step 4: Extracting PDF...")
    r, G_r = calculate_pdf(
        q=q, I_q=I_corrected,
        r_min=0.1, r_max=50.0,
        n_points=500,
        Q_damp=0.03, window="hann",
    )
    
    # Step 5: Display results
    print("\n=== Results ===")
    print(f"Q range: {q.min():.3f} - {q.max():.3f} Å⁻¹")
    print(f"r range: {r.min():.2f} - {r.max():.2f} Å")
    print(f"First peak: {r[np.argmax(G_r)]:.3f} Å")
    
    # Step 6: Save results
    print("\nSaving results...")
    np.savetxt("I_q_corrected.txt", np.column_stack([q, I_corrected]),
               header="Q (Angstrom) I(q)", fmt="%.6f %.8e")
    np.savetxt("G_r.txt", np.column_stack([r, G_r]),
               header="r (Angstrom) G(r)", fmt="%.6f %.8e")
    print("✓ Results saved to I_q_corrected.txt and G_r.txt")
    
    return q, I_corrected, r, G_r


if __name__ == "__main__":
    # Run example (with valid file paths)
    print("H5 Processor Examples")
    print("=" * 50)
    print("\nNote: Please update file paths to point to your actual data files.")
    print("\nAvailable examples:")
    print("1. example_basic_integration() - Basic I(q) integration")
    print("2. example_with_corrections() - With dark/flat/mask corrections")
    print("3. example_pdf_extraction() - PDF extraction from I(q)")
    print("4. example_full_workflow() - Complete workflow from H5 to PDF")
    
    # Uncomment to run (after setting valid file paths):
    # q, I_q = example_basic_integration()
    # q, I_corrected = example_with_corrections()
    # r, G_r = example_pdf_extraction()
    # q, I_corrected, r, G_r = example_full_workflow()
