"""Tab 1: Configure input files for H5 processing."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import streamlit as st

from xpdfsuite_ID11.app_utilities import file_picker, folder_browser, list_files, save_config

@dataclass
class H5Config:
    """Configuration pour le traitement des fichiers H5."""

    # Fichiers
    sample_file: str | None
    ref_file: str | None
    poni_file: str | None
    mask_file: str | None
    dark_file: str | None
    flat_file: str | None
    spline_file: str | None
    
    # Paramètres H5
    entry_sample: int
    entry_ref: int
    frame_sample: int | str  # int ou "mean" pour la moyenne
    frame_ref: int | str
    
    # Paramètres de traitement
    polarization: float
    q_min: float
    q_max: float
    bgscale: float


def _save_upload(uploaded, suffix: str) -> str | None:
    """Write an uploaded file to a temp file once (cached by name/size); None if no upload."""
    if uploaded is None:
        return None
    cache = st.session_state.setdefault("_upload_cache", {})
    key = (uploaded.name, uploaded.size, suffix)
    if key not in cache or not os.path.exists(cache[key]):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getvalue())
            cache[key] = f.name
    return cache[key]


def render() -> H5Config:
    """Render the configuration tab and return H5Config."""
    
    st.markdown("## ⚙️ File Configuration")
    st.markdown("Specify all input files and parameters for H5 processing.")
    
    # Initialiser les valeurs par défaut dans session_state
    defaults = {
        "entry_sample": 0,
        "entry_ref": 0,
        "frame_sample": 0,
        "frame_ref": 0,
        "polarization": 0.99,
        "q_min": 0.5,
        "q_max": 20.0,
        "bgscale": 1.0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # ===== SECTION 1: Input Files =====
    st.markdown("### 📂 Input Files")
    
    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    c1.text_input("Working folder", key="data_root")
    with c2.popover("Browse…", width='stretch'):
        folder_browser()
    
    root = st.session_state["data_root"]
    if os.path.isdir(root):
        save_config(data_root=root)
        h5_files = list_files(root)
    else:
        st.error("Folder not found")
        h5_files = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sample & Reference")
        
        # Les H5 BLISS sont des datasets virtuels vers des fichiers voisins
        # (ex: scan0001/eiger_0000.h5) : ils doivent être lus depuis leur emplacement
        # d'origine (pas d'upload), sinon HDF5 renvoie silencieusement des images nulles.
        sample_path = file_picker("Sample H5 file (required)", "h5_sample", h5_files, root)
        ref_path = file_picker("Reference H5 file (optional)", "h5_ref", h5_files, root)
        
        poni_file = st.file_uploader(
            "PONI calibration file (required)",
            type=["poni"],
            key="poni_file",
            help="Instrument calibration file for integration"
        )
    
    with col2:
        st.markdown("#### Corrections (optional)")
        
        mask_file = st.file_uploader(
            "Mask file (EDF format)",
            type=["edf"],
            key="mask_file",
            help="Mask for bad pixels"
        )
        
        dark_file = st.file_uploader(
            "Dark/Background file (EDF format)",
            type=["edf"],
            key="dark_file",
            help="Dark image for dark current subtraction"
        )
        
        flat_file = st.file_uploader(
            "Flat-field file (EDF format)",
            type=["edf"],
            key="flat_file",
            help="Flat-field image for normalization"
        )
        
        spline_file = st.file_uploader(
            "Spline correction file",
            type=["spline"],
            key="spline_file",
            help="Spline correction for detector efficiency"
        )
    
    poni_path = _save_upload(poni_file, ".poni")
    mask_path = _save_upload(mask_file, ".edf")
    dark_path = _save_upload(dark_file, ".edf")
    flat_path = _save_upload(flat_file, ".edf")
    spline_path = _save_upload(spline_file, ".spline")
    
    # ===== SECTION 2: H5 Parameters =====
    st.markdown("### 📊 H5 File Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        entry_sample = st.number_input(
            "Sample entry number",
            min_value=0,
            value=st.session_state["entry_sample"],
            key="input_entry_sample"
        )
        st.session_state["entry_sample"] = entry_sample
    
    with col2:
        if ref_path is not None:
            entry_ref = st.number_input(
                "Reference entry number",
                min_value=0,
                value=st.session_state["entry_ref"],
                key="input_entry_ref"
            )
            st.session_state["entry_ref"] = entry_ref
        else:
            st.markdown("**Reference entry #**")
            st.markdown("*(Upload reference file to enable)*")
            entry_ref = st.session_state["entry_ref"]
    
    with col3:
        frame_type_sample = st.selectbox(
            "Sample frame type",
            ("Frame index", "Mean of all frames"),
            key="frame_type_sample"
        )
        if frame_type_sample == "Frame index":
            frame_sample = st.number_input(
                "Sample frame #",
                min_value=0,
                value=0,
                key="input_frame_sample"
            )
        else:
            frame_sample = "mean"
    
    with col4:
        if ref_path is not None:
            frame_type_ref = st.selectbox(
                "Reference frame type",
                ("Frame index", "Mean of all frames"),
                key="frame_type_ref"
            )
            if frame_type_ref == "Frame index":
                frame_ref = st.number_input(
                    "Reference frame #",
                    min_value=0,
                    value=0,
                    key="input_frame_ref"
                )
            else:
                frame_ref = "mean"
        else:
            st.markdown("**Reference frame type**")
            st.markdown("*(Upload reference file to enable)*")
            frame_ref = st.session_state.get("frame_ref", "mean")
    
    # ===== SECTION 3: Processing Parameters =====
    st.markdown("### ⚙️ Processing Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        polarization = st.slider(
            "Polarization factor",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["polarization"],
            step=0.01,
            key="input_polarization"
        )
        st.session_state["polarization"] = polarization
    
    with col2:
        q_min = st.number_input(
            "Q min (Å⁻¹)",
            value=st.session_state["q_min"],
            step=0.1,
            key="input_q_min"
        )
        st.session_state["q_min"] = q_min
    
    with col3:
        q_max = st.number_input(
            "Q max (Å⁻¹)",
            value=st.session_state["q_max"],
            step=0.1,
            key="input_q_max"
        )
        st.session_state["q_max"] = q_max
    
    with col4:
        bgscale = st.number_input(
            "Background scale",
            value=st.session_state["bgscale"],
            step=0.01,
            key="input_bgscale"
        )
        st.session_state["bgscale"] = bgscale
    
    # ===== DISPLAY STATUS =====
    st.markdown("### ✅ Configuration Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.write("**Files Status:**")
        st.write(f"• Sample file: {'✅' if sample_path else '❌'}")
        st.write(f"• Reference file: {'✅' if ref_path else '❌'}")
        st.write(f"• PONI file: {'✅' if poni_path else '❌'}")
        st.write(f"• Mask file: {'✅' if mask_path else '⊘'}")
        st.write(f"• Dark file: {'✅' if dark_path else '⊘'}")
        st.write(f"• Flat file: {'✅' if flat_path else '⊘'}")
        st.write(f"• Spline file: {'✅' if spline_path else '⊘'}")
    
    with status_col2:
        st.write("**Parameters:**")
        st.write(f"• Sample: Entry #{entry_sample}, Frame {frame_sample}")
        st.write(f"• Reference: Entry #{entry_ref}, Frame {frame_ref}")
        st.write(f"• Q range: [{q_min:.2f}, {q_max:.2f}] Å⁻¹")
        st.write(f"• Polarization: {polarization:.2f}")
        st.write(f"• BGscale: {bgscale:.2f}")
    
    # Return configuration object
    return H5Config(
        sample_file=sample_path,
        ref_file=ref_path,
        poni_file=poni_path,
        mask_file=mask_path,
        dark_file=dark_path,
        flat_file=flat_path,
        spline_file=spline_path,
        entry_sample=entry_sample,
        entry_ref=entry_ref,
        frame_sample=frame_sample,
        frame_ref=frame_ref,
        polarization=polarization,
        q_min=q_min,
        q_max=q_max,
        bgscale=bgscale,
    )
