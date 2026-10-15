import streamlit as st
from xpdfsuite_ID11.utilities_process import convert_position_to_time, extract_number_of_images, compute_reaction_time_stop_flow
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from xpdfsuite_ID11 import extract_xpdf,XRDProcessor,ID11Data
import numpy as np
import tempfile
import json

import os, sys, json, glob
from pathlib import Path
import streamlit as st

CONFIG = Path.home() / ".id11_app.json"

def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except Exception:
        return {}

def save_config(**kw):
    cfg = load_config()
    cfg.update(kw)
    CONFIG.write_text(json.dumps(cfg))

@st.cache_data(ttl=60)
def list_files(root, pattern="*.h5"):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

# --- Dossier de recherche -------------------------------------------------
if "data_root" not in st.session_state:
    st.session_state["data_root"] = (
        load_config().get("data_root")
        or os.environ.get("ID11_DATA_ROOT")
        or str(Path.home())
    )

def _list_subdirs(path):
    """Liste les sous-dossiers accessibles (ignore les erreurs de permission)."""
    try:
        return sorted(
            d.name for d in Path(path).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return []

def _browse_go_parent():
    current = Path(st.session_state["browse_path"])
    st.session_state["browse_path"] = str(current.parent)

def _browse_enter_subdir():
    selected = st.session_state.get("browse_subdirs")
    if selected:
        current = Path(st.session_state["browse_path"])
        st.session_state["browse_path"] = str(current / selected)

def _browse_choose():
    # Exécuté par le callback on_click, donc AVANT que le widget
    # text_input(key="data_root") ne soit réinstancié : la modification
    # de st.session_state["data_root"] est alors autorisée.
    st.session_state["data_root"] = st.session_state["browse_path"]

def folder_browser():
    """Navigateur de dossiers natif Streamlit (ne dépend d'aucun binaire
    externe type zenity/tkinter, utilisable sur un serveur distant sans
    affichage graphique)."""
    if "browse_path" not in st.session_state:
        st.session_state["browse_path"] = st.session_state.get("data_root") or str(Path.home())

    current = Path(st.session_state["browse_path"])
    st.caption(f"📁 `{current}`")

    col_up, col_choose = st.columns(2)
    col_up.button("⬆️ Parent", key="browse_up", width='stretch', on_click=_browse_go_parent)
    col_choose.button("✅ Choisir", key="browse_choose", width='stretch', on_click=_browse_choose)

    subdirs = _list_subdirs(current)
    if subdirs:
        st.radio(
            "Sous-dossiers", subdirs, key="browse_subdirs",
            label_visibility="collapsed",
        )
        st.button("➡️ Entrer dans le dossier", key="browse_enter", on_click=_browse_enter_subdir)
    else:
        st.caption("Aucun sous-dossier")



# --- Sélecteur de fichier + champ d'affichage -------------------------------
def file_picker(label, key):
    def _on_select():
        st.session_state[f"{key}_path"] = st.session_state[f"{key}_select"] or ""

    st.selectbox(
        label, files, index=None, key=f"{key}_select",
        on_change=_on_select,
        format_func=lambda p: os.path.relpath(p, root),
        placeholder="Choisir un fichier...",
    )
    path = st.text_input("Fichier sélectionné", key=f"{key}_path",disabled=True)
    if path and not os.path.isfile(path):
        st.error("Fichier introuvable")
        return None
    return path or None


# Configure Streamlit page
st.set_page_config(
    page_title="PDF Peak Monitoring: ID11 pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("PDF peak monitoring: ID11 pipeline")


# Add CSS to style tab labels and reduce content font size
st.markdown("""
    <style>
        button[data-baseweb="tab"] {
            font-size: 16px !important;
            padding: 12px 24px !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
        }
        /* Reduce font size in tab content */
        .stTabs [role="tabpanel"] {
            font-size: 13px;
        }
        /* Reduce markdown and other text */
        [role="tabpanel"] p {
            font-size: 13px !important;
        }
        /* Reduce heading sizes */
        [role="tabpanel"] h2 {
            font-size: 18px !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        [role="tabpanel"] h3 {
            font-size: 15px !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.4rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Add stop button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Stop App", type="secondary"):
    st.success("👋 Thanks for using ID11 pipeline! Session ended.")
    st.stop()

# Create three tabs as defined above
tab1, tab2, tab3 = st.tabs(["⚙️ Configure Experiment", "📈 Extract xPDF", "📊 Time evolution "])


with tab1:



    st.markdown("## ⚙️ Configuration de l'expérience")
    # menu déroulant pour type of radiation
    option = st.selectbox(
    "Choisissez un type d'expérience :",
    ("lost_flow", "stop_flow"))

    c1, c2 = st.columns([6, 1],vertical_alignment="bottom")
    c1.text_input("Dossier de travail", key="data_root")
    with c2.popover("Parcourir…", width='stretch'):
        folder_browser()

    col_files, col_params = st.columns(2)

    with col_files:
        st.markdown("### 📂 Téléverser les fichiers")
        poni_file = st.file_uploader(
                    "Fichier poni pour la calibration de l'instrument (obligatoire)",
                    type=['poni'],
                    key="calibration file"
                    )
        poni_path = None
        if poni_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".poni") as tmp_poni:
                tmp_poni.write(poni_file.getvalue())
                poni_path = tmp_poni.name
            poni_file=poni_path
            
        mask_file = st.file_uploader(
                    "Fichier de masque (optionnel)",
                    type=['edf'],
                    key="mask file",
                )

        mask_path = None

        if mask_file is not None:
            # On crée un fichier temporaire sur le disque dur
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(mask_file.getvalue()) # On y écrit les octets du fichier téléchargé
                mask_path = tmp_file.name # On récupère le chemin absolu (ex: /tmp/tmp_xyz.edf)

    with col_params:
        st.markdown("### ⚙️ Paramètres de traitement")
    # champs pour la longueur d'onde
        polarization = st.number_input(
                "Facteur de polarisation",
                value=0.99,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                help="entrer le facteur de polarisation."
            )
        frame_binning = st.number_input(
            "Regroupemnent d'images consécutives (binning)",
            value=1,
            min_value=1,
            step=1,
            help="nb de frames consécutives moyennées dans une série")
        
        frame_step = st.number_input(
                "Décalage entre les frames d'une même série (None = pas de moyenne inter-séries)",
                value=None,
                step=1,
                help="décalage (en frames) entre deux répétitions de la série (None = pas de moyenne inter-séries)")
        
        nb_repeats = st.number_input(
            "Nombre de répétitions de la série (None = toutes celles qui rentrent dans nb_images)",
            value=None,
            step=1,
            help="nb de séries à moyenner (None = toutes celles qui rentrent dans nb_images)")

       

        root = st.session_state["data_root"]
        if os.path.isdir(root):
            save_config(data_root=root)
            files = list_files(root)
        else:
            st.error("Dossier introuvable")
            files = []
    col_sample, col_ref = st.columns(2)

    # ========== SAMPLE COLUMN ==========
    with col_sample:
        st.markdown("### 🔵 Echantillon")

        st.markdown("#### Téléverser les données de diffraction")


        #if st.button("Parcourir sample data..."):
        #    st.session_state["sample_path"] = pick_file()
        #sample_file = st.session_state.get("sample_path")
        sample_file = file_picker("Fichier de données de l'échantillon", "sample")

        

        entry_sample = st.number_input(
                    "Valeur d'entrée dans le fichiers de données de l'échantillon",
                    value=None,
                    min_value=0,
                    step=1,
                    help="For h5 files, the entry number corresponds to the dataset index (starting from 0). For EDF files, the entry number is always 0.",
                )

    # ========== REFERENCE COLUMN ==========
    with col_ref:
        st.markdown("### 🔴 Reference")

        # --- 1. Upload files ---
        st.markdown("#### Téléverser les fichiers")

        
        #if st.button("Parcourir reference data..."):
        #    st.session_state["ref_path"] = pick_file()
        #ref_file = st.session_state.get("ref_path")
        ref_file = file_picker("Fichier de données de référence", "ref")

        entry_ref = st.number_input(
                    "Valeur d'entrée dans le fichier de données de référence",
                    value=None,
                    min_value=0,
                    step=1,
                    help="For h5 files, the entry number corresponds to the dataset index (starting from 0). For EDF files, the entry number is always 0.",
                )

    if option == "lost_flow":
        st.markdown("### ⚡ Geometrie de la cellule µfluidique")
        
        col_lost_flow1, col_lost_flow2 = st.columns(2)

        with col_lost_flow1:
            # position0 tuple for the lost flow experiment
            
            position0 = st.text_input("Coordonnées de la position 0 après le mixer (format: X, Y)",
        value="0, 0",
        help="Entrez les coordonnées de la position 0 après le mixer, séparées par une virgule.")

            # Conversion de la chaîne "1, 2" en tuple (1, 2)
            try:
                position0 = tuple(int(x.strip()) for x in position0.split(","))
            except ValueError:
                st.error("Veuillez entrer deux nombres valides séparés par une virgule.")

            x_coude = st.number_input(
                "Longueur du premier canal horizontal de la cellule en mm.",
                value=31.0,
                step=0.1,
                help="Longueur du premier canal horizontal de la cellule en mm.")

            z_retour = st.number_input(
                        "Gap vertical entre les 2 canaux de la cellule en mm.",
                        value=1.0,
                        step=0.1,
                        help="Gap vertical entre les 2 canaux de la cellule en mm.")

            debit = st.number_input(
                        "Débit du flux en µL/sec.",
                        value=8.0,
                        step=0.01,
                        help="Débit du flux en µL/sec.")
     
        with col_lost_flow2:

            section = st.number_input(
                        "Section de la cellule en mm².",
                        value=2.0,
                        step=0.1,
                        help="Section de la cellule en mm².")

            section_tri = st.number_input(
                        "Section de la partie triangulaire de la cellule en mm².",
                        value=0.5*np.sqrt(5),
                        step=0.1,
                        help="Section de la partie triangulaire de la cellule en mm².")

            long_tri = st.number_input(
                        "longueur de la section triangulaire",
                        value=0.0,
                        step=0.1,
                        help="Longueur de la section triangulaire de la cellule en mm.")



####################################################################################################################

#                                           TAB 2: Extract PDF  

#####################################################################################################################

with tab2:


    st.markdown("## 📈 Extract PDF")
    # ========== DEFAULT VALUES ==========
    _default_bgscale = 1.0
    _default_qmin = 0.5
    _default_qmax = 24.5
    _default_qmaxinst = 24.5
    _default_rpoly = 0.9
    _default_lorch = True
    _default_composition = "Au"
    
    # ========== INPUT PARAMETERS SECTION ==========
    st.markdown("## 📋 Input Parameters")
    
    composition = st.text_input("Chemical composition", value=_default_composition, placeholder="e.g., Au, NaCl, Au3Cu")
    
    st.markdown("## ⚙️ Extraction Parameters")
    
    col_out1, col_out2, col_out3, col_out4 = st.columns(4)
    
    with col_out1:
        st.markdown("**Gamme de Rsouhaitée**")
        rmin = st.number_input("rmin (Å)", value=0.0, step=0.1)
        rmax = st.number_input("rmax (Å)", value=50.0, step=0.1)
        rstep = st.number_input("rstep (Å)", value=0.01, step=0.001)
    
    with col_out2:
        st.markdown("**Gamme de Q utilisée pour l'extraction**")
        
        qmin = st.number_input("qmin (Å⁻¹)", value=_default_qmin, step=0.01)
        qmax = st.number_input("qmax (Å⁻¹)", value=_default_qmax, step=0.01)
        

    with col_out3:
        st.markdown("**Paramètres pour la soustraction polynomiale**")
        qmaxinst = st.number_input("qmaxinst (Å⁻¹)", value=_default_qmaxinst, step=0.01,help="Maximum Q value for the instrument correction. If None, the maximum Q value will be used.")
        rpoly = st.number_input("rpoly", value=_default_rpoly, step=0.01, help="Polynomial correction parameter. It should be kept < 1.5.")
        lorch = st.checkbox("Lorch", value=_default_lorch, help="Apply Lorch modification to the Fourier transform. It is recommended to keep this option checked.")

    with col_out4:
        st.markdown("**Background Scaling**")
        bgscale = st.number_input("bgscale", value=_default_bgscale, step=0.01, help="Scaling factor for the background subtraction. It should be kept < 1.5.")


    if st.button("Calculate PDF", type="primary"):
        
        if sample_file is None:
            st.error("Please upload sample file ")
        else:
            # Call the appropriate function based on the selected option

            ############################################### LOST FLOW ########################################################
            if option == "lost_flow":
                if entry_sample is None:
                    eiger = ID11Data(sample_file)
                    entry_sample = eiger.entries[0]
                else:
                    eiger = ID11Data(sample_file)
                
                nb_images = extract_number_of_images(sample_file, entry_sample,poni_file)
                nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
                pdf_files = []
                if nb_images != nb_images_ref:
                    raise ValueError("The number of images in the sample and reference files do not match.")
            
                if frame_step is None:
                    # Comportement actuel : groupes consécutifs uniquement
                    nb_bins = nb_images // frame_binning
                    if nb_bins == 0:
                        raise ValueError(f"frame_binning={frame_binning} is larger than the number of images ({nb_images}).")
                    if nb_images % frame_binning != 0:
                        print(f"Warning: {nb_images} images is not a multiple of frame_binning={frame_binning}; "
                            f"the last {nb_images % frame_binning} image(s) will be discarded.")
                    nb_repeats_eff = 1
                    series_len = nb_images
                else:
                    if frame_step < frame_binning:
                        raise ValueError(f"frame_step={frame_step} must be >= frame_binning={frame_binning}.")
                    series_len = frame_step
                    max_repeats = nb_images // frame_step
                    nb_repeats_eff = max_repeats if nb_repeats is None else nb_repeats
                    if nb_repeats_eff < 1 or nb_repeats_eff > max_repeats:
                        raise ValueError(f"nb_repeats={nb_repeats} invalid: {nb_images} images with "
                                        f"frame_step={frame_step} allow at most {max_repeats} repeats.")
                    if frame_step % frame_binning != 0:
                        print(f"Warning: frame_step={frame_step} is not a multiple of frame_binning={frame_binning}; "
                            f"the last {frame_step % frame_binning} frame(s) of each series will be discarded.")
                    if nb_images % frame_step != 0 and nb_repeats is None:
                        print(f"Warning: {nb_images % frame_step} trailing image(s) do not form a complete series and will be discarded.")
                    nb_bins = frame_step // frame_binning

                st.session_state["nb_bins"] = nb_bins
                res = {}
                sample_processor = XRDProcessor(sample_file, frame=0, poni_file=poni_file,
                                                                    mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                st.session_state["samplename"] = sample_processor.samplename
                fig = go.Figure(
                    layout=go.Layout(
                        title=f"Lost flow - sample: {sample_processor.samplename}",  # Titre général
                        xaxis_title="r (Å)",
                        yaxis_title="G(r)"))
                palette = px.colors.sample_colorscale(
    "Bluered", [i / max(nb_bins - 1, 1) for i in range(nb_bins)]
)
                
                n=0
                for bin_idx in range(nb_bins):
                    couleur_courbe = palette[bin_idx % len(palette)]
                    base = bin_idx * frame_binning
                    step = frame_step if frame_step is not None else 0
                    frame_group = [
                        base + k + r * step
                        for r in range(nb_repeats_eff)
                        for k in range(frame_binning)
                    ]
                    frame_arg = frame_group[0] if len(frame_group) == 1 else frame_group
                    res[str(frame_group)] = {}
                    print(f"Processing frame group: {frame_group}")

                    sample_processor = XRDProcessor(sample_file, frame=frame_arg, poni_file=poni_file,
                                                    mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                    if ref_file is not None:
                        ref_processor = XRDProcessor(ref_file, frame=frame_arg, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
                    else:
                        ref_processor = None
            
                    
                    if len(frame_group) == 1:
                        frame_str = str(frame_group[0])
                    else:
                        frame_str = f"{frame_group[0]}-{frame_group[-1]}"
                    #outputfile = f'{os.path.dirname(sample_file)}/{os.path.basename(sample_file).split(".")[0]}_frames={frame_str}.gr'


                    # Retrieve motor positions for the current frame group and compute reaction time based on the motor positions
                    
                    scanned = sample_processor.scanned_motors

                    # Position de chaque moteur (on suppose la position constante sur les frames binnées → premier élément)
                    position = {
                        motor: value[bin_idx] if frame_binning == 1 else value[frame_group][0]
                        for motor, value in scanned.items()
                    }

                    # Vecteur [y, z]
                    position_tuple = np.array([
                        next((v for m, v in position.items() if 'y' in m), 0.0),
                        next((v for m, v in position.items() if 'z' in m), 0.0),
                    ])
                    

                    print(f"position tuple for frame group {frame_group}: {position_tuple}")
                    time = convert_position_to_time(position_tuple,position0=position0, x_coude=x_coude, z_retour=z_retour, debit=debit,
                   section=section, long_tri=long_tri, section_tri=section_tri, tol=0.1)

                    # Extract PDF from sample and reference data                                     
                    r, G = extract_xpdf(sample_processor,
                                        ref_processor=ref_processor,
                                        composition=composition,
                                        rmin=rmin,
                                        rmax=rmax,
                                        rstep=rstep,
                                        #outputfile=outputfile,
                                        interactive=False,
                                        plot=False,
                                        bgscale=bgscale,
                                        qmin=qmin,
                                        qmax=qmax,
                                        qmaxinst=qmaxinst,
                                        rpoly=rpoly)
                    # ajout plotly pour visualiser G(r) en fonction du temps
                    res[str(frame_group)]['r'] = r
                    res[str(frame_group)]['G'] = G
                    res[str(frame_group)]['time'] = time
                    
                    fig.add_trace(go.Scatter(x=r, 
                        y=G / G.max()+n, 
                        mode='lines', 
                        name=f"time={time:.2f}s",
                        line=dict( color=couleur_courbe,width=2))) # <--- On force l'application de la couleur ici
                    n+=0.02
                                    
                                    
                fig.show()
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["res"] = res   # <-- à ajouter



#################################################################" STOP FLOW ########################################"




            elif option == "stop_flow":
                if entry_sample is None:
                    
                    eiger = ID11Data(sample_file)
                    entry_sample = eiger.entries[0]
                    
                
                nb_images = extract_number_of_images(sample_file, entry_sample,poni_file)
                nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
                pdf_files = []
                if nb_images != nb_images_ref:
                    raise ValueError("The number of images in the sample and reference files do not match.")
            
                
                nb_bins = nb_images
                st.session_state["nb_bins"] = nb_bins
                res = {}
                sample_processor = XRDProcessor(sample_file, frame=0, poni_file=poni_file,
                                                                    mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                fig = go.Figure(
                    layout=go.Layout(
                        title=f"Stop flow - sample {sample_processor.samplename}",  # Titre général
                        xaxis_title="r (Å)",
                        yaxis_title="G(r)"))
                palette = px.colors.sample_colorscale("Bluered", [i / max(nb_bins - 1, 1) for i in range(nb_images)]
)
                n=0
                for bin_idx in range(nb_images):
                    couleur_courbe = palette[bin_idx % len(palette)]
                    res[str(bin_idx)] = {}
                    print(f"Processing frame group: {bin_idx}")
                    frame_arg = bin_idx
                    sample_processor = XRDProcessor(sample_file, frame=frame_arg, poni_file=poni_file,
                                                    mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                    
                    if ref_file is not None:
                        ref_processor = XRDProcessor(ref_file, frame=frame_arg, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
                    else:
                        ref_processor = None
                    frame_group = [bin_idx] 
                    # Extract PDF from sample and reference data
                    if len(frame_group) == 1:
                        frame_str = str(frame_group[0])
                    else:
                        frame_str = f"{frame_group[0]}-{frame_group[-1]}"
                    #outputfile = f'{os.path.dirname(sample_file)}/{os.path.basename(sample_file).split(".")[0]}_frames={frame_str}.gr'
                    timedict={}
            
                    reaction_time =compute_reaction_time_stop_flow(sample_processor)
                    for i, frame in enumerate(frame_group):
                        timedict[str(frame)] = reaction_time[i]
            
                    res[str(bin_idx)]['reaction_time'] = timedict
                    
                    r, G = extract_xpdf(sample_processor,
                                        ref_processor=ref_processor,
                                        composition=composition,
                                        rmin=rmin,
                                        rmax=rmax,
                                        rstep=rstep,
                                        #outputfile=outputfile,
                                        interactive=False,
                                        plot=False,
                                        bgscale=bgscale,
                                        qmin=qmin,
                                        qmax=qmax,
                                        qmaxinst=qmaxinst,
                                        rpoly=rpoly)
                    res[str(bin_idx)]['r'] = r
                    res[str(bin_idx)]['G'] = G
                    res[str(bin_idx)]['time'] = reaction_time[bin_idx]

                    print("PDF computed for reaction time: ", reaction_time[bin_idx])

                    fig.add_trace(go.Scatter(x=r, 
                                             y=G / G.max()+n, 
                                             mode='lines', 
                                             name=f"time={reaction_time[bin_idx]:.2f}s",
                                             line=dict( color=couleur_courbe,width=2)))
                    n+=0.02
                
                fig.show()
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["res"] = res   # <-- à ajouter
                
    json_res = json.dumps(st.session_state.get("res"), indent=4, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),)
    st.download_button(
        label="Download PDF data as JSON",
        data=json_res,
            file_name=f"pdf_data_sample_{st.session_state.get('samplename')}.json",
            mime="application/json",
        key="download_pdf_data"
    )
###########################################################################################################################

#                                           TAB 3: Time evolution of PDF peaks

###########################################################################################################################

with tab3:
    st.markdown("##📊 Time evolution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select the PDF peaks to monitor")
        peak_positions = st.text_input("Sélection des ppositions de pics à suivre (format: X, Y, Z...)",
                value="2.3, 2.9",
                help="Entrez les positions de pics à suivre, séparées par une virgule.")
        
        # Conversion de la chaîne "1, 2" en tuple (1, 2)
        try:
            positions = tuple(float(x.strip()) for x in peak_positions.split(","))
        except ValueError:
            st.error("Veuillez entrer deux nombres valides séparés par une virgule.")

    with col2:
        st.markdown("### Select the fitting window for each peak")
        fit_window = st.number_input("Largeur de la fenêtre de fit autour de chaque pic",
                value=0.4,
                step=0.01,
                help="Entrez la largeur de la fenêtre de fit autour de chaque pic.")

    
    ######################"  Execution button to fit peaks and plot evolution  ########################"
    
    if st.button("Fit peaks and plot evolution", type="primary"):
        res = st.session_state.get("res")
        
        if res is None:
            st.error("Please compute the PDF first in the previous tab.")
        else:
            peak_evolution = {}
            for frame_group, data in res.items():
                    r = res[frame_group]['r']
                    G = res[frame_group]['G']
                    time = res[frame_group]['time']
                    # reaction_time has the same length as the list of frame_groups, so we can use it to label the x-axis in the plot
                    time_value = time
                    peak_evolution[time_value] = {}
                    for peak_pos in positions:
                        print(f"Fitting peak at {peak_pos} Å for frame group {frame_group} (time={time_value})")
                        try:
                            popt, pcov, area = fit_peak_pseudovoigt(r, G, position=peak_pos, window=fit_window, plot=False)
                            peak_evolution[time_value][peak_pos] = {
                                'amplitude': popt[0],
                                'center': popt[1],
                                'fwhm': popt[2],
                                'eta': popt[3],
                                'bkg': popt[4],
                                'area': area
                            }
                        except Exception as e:
                            print(f"Peak fitting failed for frame group {frame_group} at peak position {peak_pos}: {e}")
                            peak_evolution[time_value][peak_pos] = None

            # make plotly plots for each peak position
            
            # Configuration des 4 sous-graphiques : (clé, titre de l'axe Y, ligne, colonne)
            panels = [
                ('amplitude', 'Peak Amplitude', 1, 1),
                ('area',      'Peak Area',      1, 2),
                ('fwhm',      'Peak FWHM',      2, 1),
                ('center',    'Peak Center',    2, 2),
            ]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[p[1] for p in panels],
                horizontal_spacing=0.1,
                vertical_spacing=0.15,
            )

            colors = px.colors.qualitative.Plotly  # une couleur par pic, cohérente entre les 4 graphiques

            for i, peak_pos in enumerate(positions):
                # Collecte des données pour ce pic
                times = []
                data = {'amplitude': [], 'area': [], 'fwhm': [], 'center': []}

                for time_value in sorted(peak_evolution.keys()):
                    peak_data = peak_evolution[time_value][peak_pos]
                    if peak_data is not None:
                        times.append(time_value)
                        for key in data:
                            data[key].append(peak_data[key])

                # Ajout d'une trace par sous-graphique
                for key, ylabel, row, col in panels:
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=data[key],
                            mode='lines+markers',
                            name=f'Peak at {peak_pos} Å',
                            legendgroup=f'peak_{peak_pos}',      # cliquer sur la légende masque le pic partout
                            showlegend=(row == 1 and col == 1),  # une seule entrée de légende par pic
                            line=dict(color=colors[i % len(colors)]),
                            marker=dict(symbol='circle'),
                        ),
                        row=row, col=col,
                    )

            # Axes : labels et grilles
            for key, ylabel, row, col in panels:
                fig.update_xaxes(title_text='Reaction Time (s)', showgrid=True, row=row, col=col)
                fig.update_yaxes(title_text=ylabel, showgrid=True, row=row, col=col)

            fig.update_layout(
                title_text='Evolution of PDF Peaks Over Time',
                height=800,
                width=1200,
                template='plotly_white',
                hovermode='x unified',
            )

            st.plotly_chart(fig, use_container_width=True)
            fig.show()
            st.session_state["peak_evolution"] = peak_evolution  # <-- à ajouter
    
    st.download_button(
        label="Download peak evolution data as JSON",
        data=json.dumps(st.session_state.get("peak_evolution"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        file_name=f"peak_evolution_data_{st.session_state.get('samplename')}.json",
        mime="application/json",
        key="download_peak_evolution_data"
    )
