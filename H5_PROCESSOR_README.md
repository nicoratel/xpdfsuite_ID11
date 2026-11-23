# H5 Processor - PDF Analysis Application

A simple Streamlit application for processing HDF5 diffraction data files and extracting Pair Distribution Functions (PDFs).

## Features

- **Tab 1 - Configure Files**: Upload and configure input files
  - Sample and reference H5 files
  - PONI calibration file for instrument geometry
  - Optional correction files (mask, dark, flat, spline)
  - Specify H5 entry numbers and frame indices
  - Set processing parameters (polarization, Q range, background scale)

- **Tab 2 - Plot I(q)**: Integrate 2D diffraction images to 1D intensity curves
  - Display sample I(q) and reference I(q)
  - Perform background subtraction with adjustable scale factor
  - Interactive plots with Plotly
  - Display I(q) statistics

- **Tab 3 - Extract PDF**: Extract G(r) with the same workflow as the monitor app
  - Uses `extract_xpdf` (composition, r range, Q range, qmaxinst, rpoly, bgscale)
  - Automatic bgscale detection (maximises the height of the first G(r) peak)
  - Open the figure in a separate browser window, export G(r) as JSON

## Installation

The application requires the xpdfsuite_ID11 package:

```bash
# Install the package (development mode recommended)
pip install -e .

# Or install with the h5 processor extras
pip install -e ".[dev]"
```

## Usage

### Command Line

Launch the application using the command-line entry point:

```bash
xpdfsuite-h5-processor
```

Optional arguments:
```bash
xpdfsuite-h5-processor --port 8503 --headless --server-address 0.0.0.0
```

Options:
- `--port PORT`: Streamlit server port (default: 8501)
- `--server-address ADDRESS`: Server address, e.g., 0.0.0.0 for network access
- `--headless`: Do not open browser automatically

### Python Script

You can also run directly with Streamlit:

```bash
streamlit run src/xpdfsuite_ID11/app_h5_processor.py
```

## Workflow

1. **Configure Files** (Tab 1):
   - Upload sample and reference H5 files
   - Upload PONI calibration file (required)
   - Optionally upload mask, dark, flat, and spline files
   - Set H5 entry numbers and frame indices
   - Adjust processing parameters

2. **Plot I(q)** (Tab 2):
   - Click "Process and Plot I(q)" button
   - View sample and reference intensity curves
   - Adjust background scale factor for reference subtraction
   - Examine I(q) statistics

3. **Extract PDF** (Tab 3):
   - Enter the chemical composition (e.g. Au)
   - Set r range (rmin, rmax, rstep), Q range (qmin, qmax), qmaxinst, rpoly, Lorch
   - Set bgscale manually, or tick "Détection automatique" and click "Calculer bgscale optimal" (needs a reference)
   - Click "Calculer PDF"
   - View G(r), open it in a separate window, download the JSON

## Processing Details

### I(q) Integration
- Loads 2D diffraction images from HDF5 files
- Applies dark current and flat-field corrections if provided
- Applies pixel mask if provided
- Integrates 2D images to 1D using PONI calibration with pyFAI
- Applies polarization factor correction

### PDF Extraction
- Re-integrates sample (and reference, if any) and calls `extract_xpdf` / `compute_xPDF` (same as the monitor app)
- Background subtraction uses `bgscale` (optionally found automatically with `auto_bgscale`)
- Polynomial correction (`qmaxinst`, `rpoly`) and Fourier transform to G(r)

## Parameters

### File Parameters
- **Sample H5 file**: Main sample data
- **Reference H5 file**: Background reference (typically empty cell or solvent)
- **PONI file**: Instrument calibration file from pyFAI
- **Mask file (optional)**: EDF file with bad pixel mask
- **Dark file (optional)**: EDF file for dark current correction
- **Flat file (optional)**: EDF file for flat-field correction
- **Spline file (optional)**: Spline correction for detector efficiency

### H5 Parameters
- **Entry number**: HDF5 entry index (useful for multi-dataset files)
- **Frame index**: Specific frame to process, or "Mean" for all frames

### Processing Parameters
- **Polarization factor**: Controls polarization correction (typically 0.9-0.95)
- **Q range**: Data range used for processing [Å⁻¹]
- **Background scale**: Scale factor for reference subtraction
- **PDF parameters** (Tab 3): composition, rmin/rmax/rstep, qmin/qmax, qmaxinst, rpoly, Lorch, bgscale

## Example Data

To test the application, you need:
1. Sample H5 file with 2D diffraction data
2. Reference H5 file with background data
3. PONI calibration file from your beamline

## Dependencies

Main dependencies:
- streamlit >= 1.30.0
- numpy >= 1.26.0
- scipy >= 1.11.0
- h5py >= 3.0.0
- pyFAI >= 2024.0.0
- plotly >= 5.0.0
- xpdfsuite_ID11 (this package)

## Troubleshooting

### "Module not found" error
Install the package in development mode:
```bash
pip install -e .
```

### HDF5 file not loading
- Check that the H5 file contains the specified entry number
- Verify the frame index is valid for the dataset
- Try opening with `h5py` to inspect the file structure

### Integration fails
- Ensure PONI file is correct for your instrument
- Check that the mask and dark files have correct dimensions
- Verify Q range parameters are reasonable

## License

This project is part of xpdfsuite_ID11.

## Author

Part of the xPDF analysis suite for ID11 experiments (ESRF).
