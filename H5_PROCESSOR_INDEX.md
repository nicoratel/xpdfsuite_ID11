# H5 Processor - Application Summary

## 📋 Project Overview

You've created a complete Streamlit application for processing HDF5 diffraction data files and extracting Pair Distribution Functions (PDFs). The application follows a 3-step workflow:

1. **Configure Files** - Upload data and set processing parameters
2. **Plot I(q)** - Integrate 2D images and perform background subtraction
3. **Extract PDF** - Calculate G(r) using Fourier transform

## 📁 Files Created

### Main Application Files

```
src/xpdfsuite_ID11/
├── app_h5_processor.py              # Main Streamlit application
├── _launcher_h5.py                  # Command-line launcher
└── tabs_h5/                         # Tab modules
    ├── __init__.py                  # Package init
    ├── configure_h5.py              # Tab 1: File configuration
    ├── plot_iq_h5.py                # Tab 2: I(q) integration & plotting
    └── extract_pdf_h5.py            # Tab 3: PDF extraction
```

### Documentation Files

```
├── H5_PROCESSOR_README.md           # Comprehensive documentation
├── H5_PROCESSOR_QUICKSTART.md       # Quick start guide
├── H5_PROCESSOR_INDEX.md            # This file
└── examples/
    └── h5_processor_example.py      # Python code examples
```

### Modified Files

```
└── pyproject.toml                   # Added entry point for launcher
```

## 🎯 Application Architecture

### Tab 1: Configure Files (`configure_h5.py`)

**Purpose**: Input files and parameter configuration

**Features**:
- File upload for sample, reference, and calibration files
- Support for optional correction files (mask, dark, flat, spline)
- H5 entry and frame number specification
- Processing parameter configuration (Q range, polarization, bgscale)
- Real-time configuration status display

**Key Class**:
- `H5Config`: Dataclass holding all configuration parameters

**Key Function**:
- `render()`: Renders the configuration UI and returns H5Config object

### Tab 2: Plot I(q) (`plot_iq_h5.py`)

**Purpose**: Integrate 2D images and plot 1D intensity curves

**Features**:
- Uses XRDProcessor from xpdfsuite_ID11 for integration
- Integrates sample and reference data using PONI calibration
- Performs background subtraction with adjustable scale factor
- Interactive Plotly plots with zoom/pan capabilities
- Statistics display (Q range, intensity maxima, correction ratios)
- Results stored in session_state for use in Tab 3

**Key Function**:
- `render(cfg)`: Processes data and displays I(q) plots

### Tab 3: Extract PDF (`extract_pdf_h5.py`)

**Purpose**: Calculate and display Pair Distribution Function

**Features**:
- Sine Fourier transform with exponential damping
- Multiple window functions (Hann, Hamming, Blackman)
- First peak detection and analysis
- G(r) statistics and peak position reporting
- CSV and TXT export capabilities
- Detailed parameter documentation

**Key Function**:
- `calculate_pdf()`: Simple PDF calculation using sine Fourier transform
- `find_peaks_simple()`: Detects peaks in G(r)
- `render(cfg)`: Displays PDF extraction UI and results

### Main Application (`app_h5_processor.py`)

**Purpose**: Top-level Streamlit application

**Features**:
- Three-tab interface
- Streamlit page configuration
- CSS styling for improved UI
- Stop button in sidebar

### Launcher (`_launcher_h5.py`)

**Purpose**: Command-line entry point

**Usage**:
```bash
xpdfsuite-h5-processor [--port PORT] [--server-address ADDRESS] [--headless]
```

**Features**:
- Argument parsing for port, server address, headless mode
- Runs the Streamlit app via CLI interface

## 🔧 Key Technologies

- **Streamlit**: Web UI framework
- **Plotly**: Interactive plotting
- **NumPy/SciPy**: Scientific computing
- **h5py**: HDF5 file handling
- **pyFAI**: 2D image integration and calibration
- **xpdfsuite_ID11**: Core processing library

## 📊 Data Flow

```
HDF5 Files
    ↓
[Tab 1: Configure]
    ↓ (H5Config object)
[Tab 2: Plot I(q)]
    ├→ XRDProcessor.integrate()
    ├→ Background subtraction
    └→ Store q, I_q, I_ref, I_corrected in session_state
    ↓
[Tab 3: Extract PDF]
    ├→ Read I_corrected from session_state
    ├→ calculate_pdf() [Fourier transform]
    └→ Display G(r) and statistics
```

## 🚀 Usage

### Installation

```bash
cd /path/to/xpdfsuite_ID11
pip install -e .
```

### Running the Application

```bash
# Using launcher
xpdfsuite-h5-processor

# Using Streamlit directly
streamlit run src/xpdfsuite_ID11/app_h5_processor.py

# Custom port
xpdfsuite-h5-processor --port 8503
```

## 📝 Key Classes and Functions

### H5Config (configure_h5.py)
```python
@dataclass
class H5Config:
    sample_file: str | None
    ref_file: str | None
    poni_file: str | None
    mask_file: str | None
    dark_file: str | None
    flat_file: str | None
    spline_file: str | None
    entry_sample: int
    entry_ref: int
    frame_sample: int | str  # int or "mean"
    frame_ref: int | str
    polarization: float
    q_min: float
    q_max: float
    bgscale: float
```

### calculate_pdf (extract_pdf_h5.py)
```python
def calculate_pdf(q, I_q, r_min, r_max, n_points, Q_damp=0.03, window=None):
    """
    Simple PDF calculation using sine Fourier transform.
    
    Returns:
        r : ndarray - r values in Å
        G : ndarray - G(r) values
    """
```

## 🎨 UI/UX Features

- **Tab-based navigation**: Logical workflow organization
- **Interactive plots**: Zoom, pan, hover information with Plotly
- **Status indicators**: ✅ for complete, ❌ for missing, ⊘ for optional
- **Parameter ranges**: Sliders and input fields with sensible defaults
- **Export buttons**: Download results as CSV or TXT
- **Error handling**: Detailed error messages with traceback expansion
- **Information sections**: Collapsible sections with explanations

## 🧪 Testing

All Python files have correct syntax and can be imported:
```bash
python -m py_compile src/xpdfsuite_ID11/app_h5_processor.py
python -m py_compile src/xpdfsuite_ID11/tabs_h5/*.py
```

## 📚 Documentation

Three levels of documentation provided:

1. **H5_PROCESSOR_README.md** - Comprehensive technical documentation
2. **H5_PROCESSOR_QUICKSTART.md** - Practical step-by-step guide
3. **examples/h5_processor_example.py** - Programmatic usage examples

## 🔄 Workflow Example

### Typical Use Case: Processing Synchrotron Data

1. **Configure** (Tab 1)
   - Upload sample H5 from beamline detector
   - Upload reference H5 (empty cell)
   - Upload PONI file from pyFAI calibration
   - Set entry numbers and frame indices
   - Configure Q range and polarization factor

2. **Plot I(q)** (Tab 2)
   - Click "Process and Plot I(q)"
   - View sample and reference curves
   - Adjust background scale slider for best subtraction
   - Inspect statistics

3. **Extract PDF** (Tab 3)
   - Set r_min=0.1, r_max=50 Å
   - Set Q_damp=0.03, window="hann"
   - Click "Extract PDF"
   - View first peak position
   - Download CSV for further analysis

## 💡 Tips for Users

- **Frame selection**: Use "mean" for averaged data with better signal-to-noise
- **Background scale**: 0.8-1.2 is typical range; adjust until corrected I(q) ≥ 0
- **Q damping**: Higher values (0.05-0.1) reduce noise but lose detail
- **Window functions**: Hann or Hamming are good defaults; Blackman for more noise suppression
- **First peak**: Typically corresponds to nearest-neighbor distances

## 🐛 Known Limitations

- Requires complete I(q) processing before PDF extraction
- PDF calculation uses simple Fourier transform (not full xPDF with composition)
- No support for masked Q regions during integration
- No peak fitting or integration tools (use external tools for this)

## 🔮 Future Enhancements

Possible additions:
- Automatic background scale optimization
- Interactive peak picking and fitting
- Multiple frame comparison
- Batch processing multiple files
- Real-time streaming data processing
- Advanced PDF deconvolution methods
- Save/load configuration presets

## 📞 Support

For issues:
1. Check documentation in `H5_PROCESSOR_README.md`
2. Review quickstart guide in `H5_PROCESSOR_QUICKSTART.md`
3. See examples in `examples/h5_processor_example.py`
4. Check HDF5 file structure with `h5py` or `HDFView`
5. Verify PONI calibration with pyFAI tools

## 📄 License

Part of xpdfsuite_ID11 project (MIT License)

## ✅ Verification Checklist

- [x] All Python files created
- [x] Syntax validated
- [x] Imports available
- [x] Entry point added to pyproject.toml
- [x] Comprehensive documentation written
- [x] Example code provided
- [x] Quick start guide created
- [x] Architecture documented

---

**Created**: 2024-11-23  
**Version**: 1.0.0  
**Status**: Ready for use ✅
