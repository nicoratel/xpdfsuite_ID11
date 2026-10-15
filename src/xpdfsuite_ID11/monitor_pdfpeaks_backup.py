import streamlit as st
from xpdfsuite_ID11.utilities_process import convert_position_to_time, extract_number_of_images, compute_reaction_time_stop_flow, match_frames_by_motor_positions
from xpdfsuite_ID11.peakfit import fit_peak_pseudovoigt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from xpdfsuite_ID11 import extract_xpdf, XRDProcessor, ID11Data
import numpy as np
import tempfile
import json
import webbrowser
import os, sys, json, glob
from pathlib import Path
import streamlit as st
from xpdfsuite_ID11.app_utilities import load_config, save_config, list_files, folder_browser, file_picker, open_fig_in_browser



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
tab1, tabiq, tabfitiq, tab2, tab3 = st.tabs(["⚙️ Configure Experiment", "📈 Plot raw data","📊 I(q) Time evolution ","🧮 Extract xPDF", "📊 G(r) Time evolution "])


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

        position_tol = st.number_input(
            "Tolérance de position pour apparier échantillon/référence (mm)",
            value=0.05,
            min_value=0.0,
            step=0.01,
            format="%.3f",
            help="Tolérance (en mm, 0.05 mm = 50 µm) sur les positions des moteurs difftz, samty et samtz "
                 "pour apparier chaque image de l'échantillon avec l'image de référence correspondante.")

       

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
        sample_file = file_picker("Fichier de données de l'échantillon", "sample", files, root)

        

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
        ref_file = file_picker("Fichier de données de référence", "ref", files, root)

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

#                                           TAB IQ: Plot raw data  

#####################################################################################################################
with tabiq:
    st.markdown("## 📈 Plot raw data")

    # ========== PLOT RAW DATA SECTION ==========
    #checkbox to substract reference data
    subtract_ref = st.checkbox("Subtract reference data", value=False)
    # slider bgscale
    bgscale = st.slider(
        "Background scaling factor",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.01,
        help="Scaling factor for the background subtraction. It should be kept < 2.0",
    )
    # save the bgscale value in session state
    st.session_state["bgscale"] = bgscale

    # ========== PLOT RAW DATA SECTION ==========
    # Appariement des frames de l'échantillon avec celles de la référence par position
    # moteur (difftz/samty/samtz), à la place de l'hypothèse "même indice de frame".
    ref_frame_map = {}
    if ref_file is not None:
        matches, unmatched = match_frames_by_motor_positions(
            sample_file, ref_file, tol=position_tol,
            entry_sample=entry_sample, entry_ref=entry_ref)
        ref_frame_map = {m['sample_frame']: m['ref_frame'] for m in matches}
        if unmatched:
            missing = [u['sample_frame'] for u in unmatched]
            st.warning(f"Aucune image de référence trouvée à {position_tol*1000:.0f} µm près pour les "
                        f"frames échantillon {missing} ; indice identique utilisé en secours pour celles-ci.")

    if st.button("Plot I(q)", type="primary"):
            
            if sample_file is None:
                st.error("Please upload sample file ")
                
            else:
                # Call the appropriate function based on the selected option
                nb_images = extract_number_of_images(sample_file, entry_sample, poni_file)
                # Lost flow experiment
                if option == "lost_flow":
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
                    
                    sample_processor = XRDProcessor(sample_file, frame=0, poni_file=poni_file,
                                                                        mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                    st.session_state["samplename"] = sample_processor.samplename
                    palette = px.colors.sample_colorscale(
        "Bluered", [i / max(nb_bins - 1, 1) for i in range(nb_bins)]
    )
                    
                    # Ne stocke ici que les données brutes (qsample, Isample, Iref) par bin, sans
                    # calculer Iplot ni construire la figure : le tracé est rejoué plus bas (slider
                    # bgscale, case "subtract_ref") sans refaire les intégrations coûteuses.
                    bins_data = []
                    progress_bar_iq_lostflow = st.progress(0, text="Intégration des frames en cours...")
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
                        
                        print(f"Integrating frame group: {frame_group}")
    
                        sample_processor = XRDProcessor(sample_file, frame=frame_arg, poni_file=poni_file,
                                                        mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                        qsample, Isample = sample_processor.integrate()

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

                        # If reference file is provided, integrate the corresponding frame available in matches
                        Iref = None
                        if ref_file is not None:
                            frame_group_ref = [ref_frame_map.get(f, f) for f in frame_group]
                            frame_arg_ref = frame_group_ref[0] if len(frame_group_ref) == 1 else frame_group_ref
                            ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
                            qref, Iref = ref_processor.integrate()

                        bins_data.append({
                            "time": time,
                            "qsample": qsample,
                            "Isample": Isample,
                            "Iref": Iref,
                            "color": couleur_courbe,
                        })
                        progress_bar_iq_lostflow.progress((bin_idx + 1) / nb_bins, text=f"Intégration {bin_idx + 1}/{nb_bins}")

                    progress_bar_iq_lostflow.empty()
                    st.session_state["iq_raw_data"] = {
                        "option": "lost_flow",
                        "samplename": sample_processor.samplename,
                        "bins": bins_data,
                    }
                
                # ======================+ STOP FLOW =========================   
                elif option == "stop_flow":
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
                    
                    sample_processor = XRDProcessor(sample_file, frame=0, poni_file=poni_file,
                                                                        mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                    st.session_state["samplename"] = sample_processor.samplename
                    palette = px.colors.sample_colorscale(
        "Bluered", [i / max(nb_bins - 1, 1) for i in range(nb_bins)]
    )
                    
                    # Idem lost_flow : seules les données brutes sont stockées ici (pas de figure).
                    bins_data = []
                    progress_bar_iq_stopflow = st.progress(0, text="Intégration des frames en cours...")
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
                        #res[str(frame_group)] = {}
                        print(f"Integrating frame group: {frame_group}")
    
                        sample_processor = XRDProcessor(sample_file, frame=frame_arg, poni_file=poni_file,
                                                        mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                        qsample, Isample = sample_processor.integrate()


                        # Retrieve time info from sample_processor epoch metadata
                        reaction_time = compute_reaction_time_stop_flow(sample_processor)
                        time = reaction_time[bin_idx] if reaction_time is not None else None

                        # If reference file is provided, integrate the corresponding frame available in matches
                        Iref = None
                        if ref_file is not None:
                            frame_group_ref = [ref_frame_map.get(f, f) for f in frame_group]
                            frame_arg_ref = frame_group_ref[0] if len(frame_group_ref) == 1 else frame_group_ref
                            ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
                            qref, Iref = ref_processor.integrate()

                        bins_data.append({
                            "time": time,
                            "qsample": qsample,
                            "Isample": Isample,
                            "Iref": Iref,
                            "color": couleur_courbe,
                        })
                        progress_bar_iq_stopflow.progress((bin_idx + 1) / nb_bins, text=f"Intégration {bin_idx + 1}/{nb_bins}")

                    progress_bar_iq_stopflow.empty()
                    st.session_state["iq_raw_data"] = {
                        "option": "stop_flow",
                        "samplename": sample_processor.samplename,
                        "bins": bins_data,
                    }


    # ========== TRACÉ (peu coûteux, rejoué à chaque interaction du slider/case à cocher) ==========
    # Reconstruit la figure à partir des données brutes mises en cache (pas de nouvelle intégration) :
    # Iplot = Isample - bgscale * Iref est recalculé ici avec la valeur courante du slider.
    iq_raw_data = st.session_state.get("iq_raw_data")
    if iq_raw_data and iq_raw_data.get("bins"):
        label_option = "Lost flow" if iq_raw_data["option"] == "lost_flow" else "Stop flow"
        fig_title = f"I(q)-{st.session_state.get('samplename')}-{label_option}"

        fig = go.Figure(
            layout=go.Layout(
                title=fig_title,
                xaxis_title="q ($Å^{-1}$)",
                yaxis_title="I(q)",
            )
        )
        for bin_data in iq_raw_data["bins"]:
            qsample = bin_data["qsample"]
            Isample = bin_data["Isample"]
            Iref = bin_data["Iref"]
            time = bin_data["time"]
            if subtract_ref and Iref is not None:
                Iplot = Isample - bgscale * Iref
            else:
                Iplot = Isample
            time_label = f"t= {time:.2f} s" if time is not None else "t= N/A"
            fig.add_trace(go.Scatter(
                x=qsample, y=Iplot, mode="lines",
                line=dict(color=bin_data["color"]), name=time_label,
            ))

        st.plotly_chart(fig, use_container_width=True)

        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_iq"):
            safe_name = f"iq_{label_option}_{iq_raw_data['samplename']}".replace(" ", "_")
            open_fig_in_browser(fig, tab_title=fig_title, filename=f"{safe_name}.html")
    st.download_button(
    label="Download I(q) data as JSON",
    data=json.dumps(st.session_state.get("iq_raw_data"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
    file_name=f"IQ_peak_evolution_data_{st.session_state.get('samplename')}.json",
    mime="application/json",
    key="download_iq_data"
)


####################################################################################################################

#                                           TAB 3: Fit I(q) peaks 

#####################################################################################################################
with tabfitiq:
    st.markdown("## 📊 I(q) Time evolution")
    iq_raw_data = st.session_state.get("iq_raw_data")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select the I(q) peaks to monitor")
        
        peak_positions = st.text_input("Sélection des ppositions de pics à suivre (format: X, Y, Z...)",
                value="1.0, 2.0",
                help="Entrez les positions de pics à suivre, séparées par une virgule.")
        
        # Conversion de la chaîne "1, 2" en tuple (1, 2)
        try:
            positions = tuple(float(x.strip()) for x in peak_positions.split(","))
        except ValueError:
            st.error("Veuillez entrer deux nombres valides séparés par une virgule.")

    with col2:
        st.markdown("### Select the fitting window for each peak")
        fit_windowiq = st.number_input("Largeur de la fenêtre dq de fit autour de chaque pic",
                value=0.4,
                step=0.01,
                help="Entrez la largeur de la fenêtre de fit (deltaQ) autour de chaque pic.")

    
    ######################"  Execution button to fit peaks and plot evolution  ########################"
    
    if st.button("Fit I(q) peaks and plot evolution", type="primary"):

        iq_raw_data = st.session_state.get("iq_raw_data")

        if iq_raw_data is None:
            st.error("Please plot I(q) data first in the previous tab.")
        else:
            peak_evolution = {}

            bins = iq_raw_data["bins"]
            total = len(bins) * len(positions)
            done = 0

            progress_bar = st.progress(0.0, text="Starting peak fitting...")

            for bin_data in bins:
                qsample = bin_data["qsample"]
                Isample = bin_data["Isample"]
                time_value = bin_data["time"]
                # reaction_time has the same length as the list of frame_groups,
                # so we can use it to label the x-axis in the plot
                peak_evolution[time_value] = {}

                for peak_pos in positions:
                    try:
                        popt, pcov, area = fit_peak_pseudovoigt(
                            qsample, Isample, position=peak_pos,
                            window=fit_windowiq, plot=False
                        )
                        peak_evolution[time_value][peak_pos] = {
                            'amplitude': popt[0],
                            'center': popt[1],
                            'fwhm': popt[2],
                            'eta': popt[3],
                            'bkg': popt[4],
                            'area': area,
                        }
                    except Exception as e:
                        print(f"Peak fitting failed for time {time_value} at peak position {peak_pos}: {e}")
                        peak_evolution[time_value][peak_pos] = None

                    done += 1
                    progress_bar.progress(
                        done / total,
                        text=f"Fitting peaks... {done}/{total} (t = {time_value} s, peak {peak_pos} Å)"
                    )

            progress_bar.empty()  # retire la barre une fois terminé
            st.session_state["peak_evolution"] = peak_evolution

            # make plotly plots for each peak position
            
            # Configuration des 4 sous-graphiques : (clé, titre de l'axe Y, ligne, colonne)
            panels = [
                ('amplitude', 'Peak Amplitude', 1, 1),
                ('area',      'Peak Area',      1, 2),
                ('fwhm',      'Peak FWHM',      2, 1),
                ('center',    'Peak Center',    2, 2),
            ]

            fig2 = make_subplots(
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
                    fig2.add_trace(
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
                fig2.update_xaxes(title_text='Reaction Time (s)', showgrid=True, row=row, col=col)
                fig2.update_yaxes(title_text=ylabel, showgrid=True, row=row, col=col)

            fig2.update_layout(
                title_text='Evolution of PDF Peaks Over Time',
                height=800,
                width=1200,
                template='plotly_white',
                hovermode='x unified',
            )

            st.session_state["fig2_iq_peaks"] = fig2

    fig2_display = st.session_state.get("fig2_iq_peaks")
    if fig2_display is not None:
        st.plotly_chart(fig2_display, use_container_width=True)
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_iq_peaks"):
            html = fig2_display.to_html(
                full_html=True,
                include_plotlyjs=True, 
                config={"responsive": True},
            )
            # Injection du titre de l'onglet
            fig_title = f"Fit pics I(Q):{st.session_state.get('samplename', 'Unknown')} {st.session_state.get('iq_raw_data')['option']} "
            open_fig_in_browser(fig2_display, tab_title=fig_title, filename=f"fit_peaks_iq_{st.session_state.get('samplename', 'Unknown')}.html")

    st.download_button(
        label="Download I(q) peak evolution data as JSON",
        data=json.dumps(st.session_state.get("peak_evolution"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        file_name=f"IQ_peak_evolution_data_{st.session_state.get('samplename')}.json",
        mime="application/json",
        key="download_iq_peak_evolution_data"
    )


####################################################################################################################

#                                           TAB 4: Extract PDF  

#####################################################################################################################

with tab2:


    st.markdown("## 🧮 Extract PDF")
    # ========== DEFAULT VALUES ==========
    _default_bgscale = st.session_state.get("bgscale", 1.0)
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


    if st.button("Calcuer PDF", type="primary"):
        
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
                #nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
                pdf_files = []
                #if nb_images != nb_images_ref:
                #    raise ValueError("The number of images in the sample and reference files do not match.")

                # Appariement des frames de l'échantillon avec celles de la référence par position
                # moteur (difftz/samty/samtz), à la place de l'hypothèse "même indice de frame".
                ref_frame_map = {}
                if ref_file is not None:
                    matches, unmatched = match_frames_by_motor_positions(
                        sample_file, ref_file, tol=position_tol,
                        entry_sample=entry_sample, entry_ref=entry_ref)
                    ref_frame_map = {m['sample_frame']: m['ref_frame'] for m in matches}
                    if unmatched:
                        missing = [u['sample_frame'] for u in unmatched]
                        st.warning(f"Aucune image de référence trouvée à {position_tol*1000:.0f} µm près pour les "
                                   f"frames échantillon {missing} ; indice identique utilisé en secours pour celles-ci.")
            
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
                progress_bar_gr_lostflow = st.progress(0, text="Calcul du PDF en cours...")
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
                        frame_group_ref = [ref_frame_map.get(f, f) for f in frame_group]
                        frame_arg_ref = frame_group_ref[0] if len(frame_group_ref) == 1 else frame_group_ref
                        ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
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
                    progress_bar_gr_lostflow.progress((bin_idx + 1) / nb_bins, text=f"Calcul du PDF {bin_idx + 1}/{nb_bins}")
                                    
                                    
                progress_bar_gr_lostflow.empty()
                st.session_state["fig_gr"] = fig
                st.session_state["res"] = res   # <-- à ajouter



#################################################################" STOP FLOW ########################################"




            elif option == "stop_flow":
                if entry_sample is None:
                    
                    eiger = ID11Data(sample_file)
                    entry_sample = eiger.entries[0]
                    
                
                nb_images = extract_number_of_images(sample_file, entry_sample,poni_file)
                #nb_images_ref = nb_images # here we assume the reference file has the same number of images as the sample file. If not, you can modify this to extract the number of images from the reference file as well.
                pdf_files = []
                #if nb_images != nb_images_ref:
                #    raise ValueError("The number of images in the sample and reference files do not match.")

                # Appariement des frames de l'échantillon avec celles de la référence par position
                # moteur (difftz/samty/samtz), à la place de l'hypothèse "même indice de frame".
                ref_frame_map = {}
                if ref_file is not None:
                    matches, unmatched = match_frames_by_motor_positions(
                        sample_file, ref_file, tol=position_tol,
                        entry_sample=entry_sample, entry_ref=entry_ref)
                    ref_frame_map = {m['sample_frame']: m['ref_frame'] for m in matches}
                    if unmatched:
                        missing = [u['sample_frame'] for u in unmatched]
                        st.warning(f"Aucune image de référence trouvée à {position_tol*1000:.0f} µm près pour les "
                                   f"frames échantillon {missing} ; indice identique utilisé en secours pour celles-ci.")
            
                
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
                progress_bar_gr_stopflow = st.progress(0, text="Calcul du PDF en cours...")
                for bin_idx in range(nb_images):
                    couleur_courbe = palette[bin_idx % len(palette)]
                    res[str(bin_idx)] = {}
                    print(f"Processing frame group: {bin_idx}")
                    frame_arg = bin_idx
                    sample_processor = XRDProcessor(sample_file, frame=frame_arg, poni_file=poni_file,
                                                    mask=mask_path, polarization_factor=polarization, entry=entry_sample)
                    
                    if ref_file is not None:
                        frame_arg_ref = ref_frame_map.get(bin_idx, bin_idx)
                        ref_processor = XRDProcessor(ref_file, frame=frame_arg_ref, poni_file=poni_file, mask=mask_path, polarization_factor=polarization, entry=entry_ref)
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
                    progress_bar_gr_stopflow.progress((bin_idx + 1) / nb_images, text=f"Calcul du PDF {bin_idx + 1}/{nb_images}")
                
                progress_bar_gr_stopflow.empty()
                st.session_state["fig_gr"] = fig
                st.session_state["res"] = res   # <-- à ajouter

    fig_gr_display = st.session_state.get("fig_gr")
    if fig_gr_display is not None:
        st.plotly_chart(fig_gr_display, use_container_width=True)
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_gr"):
            
            html = fig_gr_display.to_html(
                full_html=True,
                include_plotlyjs=True, 
                config={"responsive": True},
            )
            # Injection du titre de l'onglet
            fig_title = f"G(r):{st.session_state.get('samplename', 'Unknown')} {st.session_state.get('iq_raw_data')['option']} "
            open_fig_in_browser(fig_gr_display, tab_title=fig_title, filename=f"PDFs_{st.session_state.get('samplename', 'Unknown')}.html")

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
    st.markdown("##📊 PDF peaks Time evolution")

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
    
    if st.button("Fit PDF peaks and plot evolution", type="primary"):
        res = st.session_state.get("res")
        
        if res is None:
            st.error("Please compute the PDF first in the previous tab.")
        else:
            peak_evolution = {}
            nb_frame_groups = len(res)
            progress_bar_peaks = st.progress(0, text="Fit des pics en cours...")
            for group_idx, (frame_group, data) in enumerate(res.items()):
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
                    progress_bar_peaks.progress((group_idx + 1) / nb_frame_groups, text=f"Fit des pics {group_idx + 1}/{nb_frame_groups}")

            progress_bar_peaks.empty()
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

            st.session_state["fig_pdf_peaks"] = fig
            st.session_state["peak_evolution"] = peak_evolution  # <-- à ajouter

    fig_pdf_peaks_display = st.session_state.get("fig_pdf_peaks")
    if fig_pdf_peaks_display is not None:
        st.plotly_chart(fig_pdf_peaks_display, use_container_width=True)
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_pdf_peaks"):
            
                
            html = fig_pdf_peaks_display.to_html(
                full_html=True,
                include_plotlyjs=True, 
                config={"responsive": True},
            )
            # Injection du titre de l'onglet
            fig_title = f"Fit pics G(r):{st.session_state.get('samplename', 'Unknown')} {st.session_state.get('iq_raw_data')['option']} "
            open_fig_in_browser(fig_pdf_peaks_display, tab_title=fig_title, filename=f"fit_peaks_gr_{st.session_state.get('samplename', 'Unknown')}.html")
    st.download_button(
        label="Download PDF peak evolution data as JSON",
        data=json.dumps(st.session_state.get("peak_evolution"), indent=4,default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        file_name=f"PDF_peak_evolution_data_{st.session_state.get('samplename')}.json",
        mime="application/json",
        key="download_pdf_peak_evolution_data"
    )
