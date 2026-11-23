# H5 Processor - Quick Start Guide

## Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install xpdfsuite_ID11

From the repository root:

```bash
# Using pip in development mode
pip install -e .
```

This will install all dependencies including:
- streamlit
- h5py
- pyFAI
- numpy, scipy
- plotly
- And other scientific packages

### 3. Verify Installation

```bash
python -c "from xpdfsuite_ID11 import XRDProcessor; print('✓ Installation successful')"
```

## Launching the Application

### Method 1: Using Command-line Launcher (Recommended)

```bash
xpdfsuite-h5-processor
```

This opens the app in your default browser at `http://localhost:8501`

### Method 2: Using Streamlit Directly

```bash
streamlit run src/xpdfsuite_ID11/app_h5_processor.py
```

### Method 3: Custom Port

```bash
xpdfsuite-h5-processor --port 8503
```

## Preparing Your Data

Before running the app, gather:

1. **Sample H5 file** - Main diffraction data
   - Format: HDF5 with 2D diffraction images
   - Entry: Usually labeled 'entry' or similar (configure in Tab 1)
   - Frame: Select specific frame or use mean of all frames

2. **Reference H5 file** (optional) - Background/empty cell data
   - Same format as sample file
   - Used for background subtraction; leave empty to plot I(q) without subtraction

3. **PONI Calibration File** (Required)
   - Generated from pyFAI calibration: `pyFAI-calib`
   - Contains instrument geometry (detector distance, center, tilt, etc.)
   - Essential for I(q) integration

4. **Optional Correction Files**:
   - **Mask file** (EDF): Marks bad pixels (0=use, non-zero=ignore)
   - **Dark file** (EDF): Dark current/background image
   - **Flat-field file** (EDF): Detector flat-field response
   - **Spline file**: Detector efficiency correction

## Workflow: Step-by-Step

### Tab 1: Configure Files

1. **Select Files**
   - Type the local path of the sample H5 file (not an upload: BLISS/ID11 master files are virtual datasets linked to neighbouring raw files such as `scan0001/eiger_0000.h5`, which must stay next to it, otherwise images are read as all zeros)
   - (Optional) Type the path of the reference H5 file for background subtraction
   - Click "PONI file" and select calibration file
   - (Optional) Upload mask, dark, flat-field files

2. **Set H5 Parameters**
   - **Entry #**: Usually 1 (check your HDF5 file structure)
   - **Frame #**: Frame index (e.g., 0, 5, 10) or "Mean of all frames"

3. **Set Processing Parameters**
   - **Polarization factor**: 0.90-0.95 (typical for X-rays)
   - **Q range**: [0.5, 20.0] Å⁻¹ (adjust based on your detector)
   - **Background scale**: Initially 1.0 (adjust in Tab 2)

4. **Check Status**
   - Green checkmarks (✅) indicate ready
   - Circles (⊘) indicate optional files not provided

### Tab 2: Plot I(q)

1. **Configure Processing**
   - Toggle "Subtract reference data" to on/off
   - Adjust "Background scale factor" slider (0.0 - 2.0)
   - Higher values → more background subtracted

2. **Process**
   - Click "Process and Plot I(q)" button
   - Wait for integration (usually 10-30 seconds)
   - View I(q) curves with Plotly interactive plots

3. **Inspect Results**
   - Zoom, pan, hover over data points
   - Check I(q) statistics
   - If background subtraction is incorrect, go back and adjust scale

### Tab 3: Extract PDF

1. **Configure PDF Parameters**
   - **Chemical composition**: e.g. `Au`, `NaCl`
   - **rmin / rmax / rstep**: real-space range (Å)
   - **qmin / qmax**: Q range used for the Fourier transform
   - **qmaxinst / rpoly**: polynomial background correction (keep rpoly < 1.5)
   - **bgscale**: manual value, or "Détection automatique" then "Calculer bgscale optimal" (requires a reference file)

2. **Extract PDF**
   - Click "Calculer PDF"
   - View G(r); "Ouvrir dans une fenêtre séparée" opens it in the browser

3. **Export Results**
   - Download the JSON with r and G(r)

## Troubleshooting

### "PONI file not found"
- Ensure PONI file is uploaded (required for integration)
- Check file format: should be a text file with .poni extension

### "Entry not found in H5 file"
- Entry numbers are 0-indexed in HDF5
- Inspect your H5 file: `h5dump -H sample.h5` or use `h5py`
- Common entries: 0, 1, or specific beamline names

### "Integration produces all zeros"
- Check detector is not saturated
- Verify PONI calibration is correct
- Try a known good frame first
- Check mask file (should mostly be 0s)

### "PDF extraction fails"
- Check the sample/reference entry and frame in Tab 1
- Check the chemical composition is valid
- Check qmax / qmaxinst do not exceed the measured Q range

### "Background subtraction looks wrong"
- Adjust bgscale factor in Tab 2
- Use slider to find optimal value
- Usually 0.8-1.2 is reasonable range
- Watch the corrected I(q) for negative values (indicates over-subtraction)

## Tips & Tricks

1. **Background Scale**
   - Use "Détection automatique" in Tab 3 to find the bgscale maximising the first G(r) peak

2. **Q Range**
   - Minimum Q: Usually limited by beamstop or geometry (~0.3-1.0 Å⁻¹)
   - Maximum Q: Limited by detector (check your detector specs)
   - Lower Q_max = shorter r_max achievable

## Example Workflows

### Single-frame processing
1. Upload single H5 frame
2. Set entry and frame number
3. Process and extract PDF

### Averaging frames
1. Upload H5 with multiple frames
2. Select "Mean of all frames" in Tab 1
3. Process and extract PDF (improves signal-to-noise)

### Optimization workflow
1. Process different bgscale values in Tab 2
2. Find optimal value where corrected curve has no negative regions
3. Extract PDF with optimized value

## Advanced Usage

See `examples/h5_processor_example.py` for programmatic API usage:

```bash
python examples/h5_processor_example.py
```

## Support

For issues or questions:
1. Check file paths and formats
2. Inspect data with HDFView or h5py
3. Verify PONI calibration with pyFAI
4. See detailed error messages in browser console (F12)

## References

- [pyFAI Documentation](https://pyfai.readthedocs.io/)
- [HDF5 Format](https://www.h5py.org/)
- [PDF Analysis](https://en.wikipedia.org/wiki/Pair_distribution_function)
- [ESRF ID11 Beamline](https://www.esrf.eu/)
