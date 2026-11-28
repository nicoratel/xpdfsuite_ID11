"""Onglet « Extract xPDF » : extraction des PDF G(r) pour chaque frame/groupe de frames."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from xpdfsuite_ID11 import extract_xpdf, ID11Data, XRDProcessor
from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.pdf_extraction import auto_bgscale
from xpdfsuite_ID11.utilities_process import (
    compute_reaction_time_stop_flow,
    convert_position_to_time,
    extract_number_of_images,
    match_frames_by_motor_positions,
)

from .configure import ExperimentConfig
position_tol=0.05

def render(cfg: ExperimentConfig) -> None:
    # --- Paramètres fournis par l'onglet « Configure Experiment » ---
    sample_file = cfg.sample_file
    ref_file = cfg.ref_file
    poni_file = cfg.poni_file
    mask_path = cfg.mask_path
    dark = cfg.dark
    flat = cfg.flat
    spline = cfg.spline
    polarization = cfg.polarization
    entry_sample = cfg.entry_sample
    entry_ref = cfg.entry_ref
    frame_binning = cfg.frame_binning
    frame_step = cfg.frame_step
    #nb_repeats = cfg.nb_repeats
    option = cfg.option
    
    position0 = cfg.position0
    

    st.markdown("## 🧮 Extract PDF")
    # ========== DEFAULT VALUES ==========
    _default_bgscale = st.session_state.get("bgscale", 1.0)
    _default_qmin = 0.5
    _default_qmax = 20.0
    _default_qmaxinst = 20.0
    _default_rpoly = 0.9
    _default_lorch = True
    _default_composition = "Au"
    
    # ========== INPUT PARAMETERS SECTION ==========
    st.markdown("## 📋 Input Parameters")

    if "composition" not in st.session_state:
        st.session_state["composition"] = _default_composition
    composition = st.text_input(
        "Chemical composition", placeholder="e.g., Au, NaCl, Au3Cu", key="composition",
    )

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
        auto_bgscale_enabled = st.checkbox(
            "🔍 Détection automatique (1er pic PDF)",
            value=st.session_state.get("auto_bgscale_enabled", False),
            key="auto_bgscale_enabled",
            help="Recherche le bgscale qui maximise la hauteur du premier pic de G(r).",
        )

        if auto_bgscale_enabled:
            r_peak_auto = st.number_input(
                "r du 1er pic (Å)",
                value=st.session_state.get("r_peak_auto", 2.8),
                step=0.1,
                key="r_peak_auto",
            )
            if st.button("🔄 Calculer bgscale optimal", key="btn_auto_bgscale_extract"):
                if sample_file is None or ref_file is None:
                    st.error("Fichiers échantillon et référence requis pour la détection automatique.")
                else:
                    with st.spinner("Calcul du bgscale optimal..."):
                        sample_processor0 = XRDProcessor(
                            sample_file, frame=0, poni_file=poni_file, mask=mask_path,
                            dark=dark, flat=flat, spline=spline,
                            polarization_factor=polarization, entry=entry_sample,
                        )
                        ref_processor0 = XRDProcessor(
                            ref_file, frame=0, poni_file=poni_file, mask=mask_path,
                            dark=dark, flat=flat, spline=spline,
                            polarization_factor=polarization, entry=entry_ref,
                        )
                        q0, I0 = sample_processor0.integrate()
                        q0ref, Iref0 = ref_processor0.integrate()
                        Iref0 = np.interp(q0, q0ref, Iref0)
                        try:
                            bg_opt, h_opt, r_found = auto_bgscale(
                                q0, I0, Iref0, composition,
                                qmin=qmin, qmax=qmax, qmaxinst=qmaxinst,
                                rpoly=rpoly, Lorch=lorch, r_peak=r_peak_auto,
                            )
                            st.session_state["bgscale_value"] = bg_opt
                            st.success(
                                f"bgscale optimal = {bg_opt:.4f} "
                                f"(pic G(r) = {h_opt:.3f} à r = {r_found:.2f} Å)"
                            )
                        except Exception as exc:
                            st.error(f"Échec de la détection automatique : {exc}")

        if "bgscale_value" not in st.session_state:
            st.session_state["bgscale_value"] = _default_bgscale
        bgscale = st.number_input(
            "bgscale", step=0.01, key="bgscale_value",
            help="Scaling factor for the background subtraction. It should be kept < 1.5.",
        )
        st.session_state["bgscale"] = bgscale


    if st.button("Calculer PDF", type="primary"):
        
        if sample_file is None:
            st.error("Please upload sample file ")
        else:
            # Call the appropriate function based on the selected option

            ############################################### LOST FLOW ########################################################
            
            if option == "lost_flow":

                # ---------------------------------------------------------
                # 1. Initialisation des données et de la référence
                # ---------------------------------------------------------
                if entry_sample is None:
                    eiger = ID11Data(sample_file)
                    entry_sample = eiger.entries[0]
                else:
                    eiger = ID11Data(sample_file)

                nb_images = extract_number_of_images(
                    sample_file, entry_sample, poni_file
                )

                pdf_files = []

                # Appariement des frames de l'échantillon avec celles de la
                # référence à partir des positions des moteurs
                ref_frame_map = {}

                if ref_file is not None:
                    matches, unmatched = match_frames_by_motor_positions(
                        sample_file,
                        ref_file,
                        tol=position_tol,
                        entry_sample=entry_sample,
                        entry_ref=entry_ref,
                    )

                    ref_frame_map = {
                        m["sample_frame"]: m["ref_frame"]
                        for m in matches
                    }

                    if unmatched:
                        missing = [u["sample_frame"] for u in unmatched]
                        st.warning(
                            f"Aucune image de référence trouvée à "
                            f"{position_tol * 1000:.0f} µm près pour les "
                            f"frames échantillon {missing} ; indice identique "
                            f"utilisé en secours pour celles-ci."
                        )

                # ---------------------------------------------------------
                # 2. Définition du pas et calcul du nombre de groupes
                # ---------------------------------------------------------
                if frame_binning < 1:
                    raise ValueError("frame_binning must be >= 1.")

                if frame_step is None:
                    # Groupes consécutifs
                    frame_step_eff = frame_binning
                else:
                    # frame_step = décalage entre les débuts des groupes
                    if frame_step < frame_binning:
                        raise ValueError(
                            f"frame_step={frame_step} must be >= "
                            f"frame_binning={frame_binning}."
                        )

                    frame_step_eff = frame_step

                # Nombre de groupes complets possibles
                if nb_images < frame_binning:
                    nb_bins = 0
                else:
                    nb_bins = (
                        (nb_images - frame_binning) // frame_step_eff + 1
                    )

                if nb_bins == 0:
                    raise ValueError(
                        f"frame_binning={frame_binning} is larger than "
                        f"the number of images ({nb_images})."
                    )

                # Vérification des images restantes en fin de séquence
                n_images_used = (
                    (nb_bins - 1) * frame_step_eff + frame_binning
                )

                if n_images_used < nb_images:
                    print(
                        f"Warning: {nb_images - n_images_used} trailing "
                        f"image(s) will not be used."
                    )

                if frame_step_eff > frame_binning:
                    print(
                        f"Info: {frame_step_eff - frame_binning} image(s) "
                        f"will be skipped between consecutive groups."
                    )

                st.session_state["nb_bins"] = nb_bins

                # ---------------------------------------------------------
                # 3. Initialisation du processeur et de la figure
                # ---------------------------------------------------------
                res = {}

                sample_processor = XRDProcessor(
                    sample_file,
                    frame=0,
                    poni_file=poni_file,
                    mask=mask_path,
                    dark=dark,
                    flat=flat,
                    spline=spline,
                    polarization_factor=polarization,
                    entry=entry_sample,
                )

                st.session_state["samplename"] = sample_processor.samplename

                
                progress_bar_gr_lostflow = st.progress(
                    0,
                    text="Calcul de la PDF en cours...",
                )

                # ---------------------------------------------------------
                # 4. Calcul du PDF pour chaque groupe d'images
                # ---------------------------------------------------------
                for bin_idx in range(nb_bins):

                    #couleur_courbe = palette[bin_idx % len(palette)]

                    # Début du groupe courant
                    base = bin_idx * frame_step_eff

                    # Frames moyennées / intégrées dans ce groupe
                    frame_group = [
                        base + k for k in range(frame_binning)
                    ]

                    frame_arg = (
                        frame_group[0]
                        if len(frame_group) == 1
                        else frame_group
                    )

                    res[str(frame_group)] = {}

                    print(f"Processing frame group: {frame_group}")

                    sample_processor = XRDProcessor(
                        sample_file,
                        frame=frame_arg,
                        poni_file=poni_file,
                        mask=mask_path,
                        dark=dark,
                        flat=flat,
                        spline=spline,
                        polarization_factor=polarization,
                        entry=entry_sample,
                    )

                    # -----------------------------------------------------
                    # 5. Chargement du groupe de référence correspondant
                    # -----------------------------------------------------
                    if ref_file is not None:
                        frame_group_ref = [
                            ref_frame_map.get(f, f)
                            for f in frame_group
                        ]

                        frame_arg_ref = (
                            frame_group_ref[0]
                            if len(frame_group_ref) == 1
                            else frame_group_ref
                        )

                        ref_processor = XRDProcessor(
                            ref_file,
                            frame=frame_arg_ref,
                            poni_file=poni_file,
                            mask=mask_path,
                            dark=dark,
                            flat=flat,
                            spline=spline,
                            polarization_factor=polarization,
                            entry=entry_ref,
                        )
                    else:
                        ref_processor = None

                    # Nom de la séquence de frames
                    if len(frame_group) == 1:
                        frame_str = str(frame_group[0])
                    else:
                        frame_str = (
                            f"{frame_group[0]}-{frame_group[-1]}"
                        )

                    # -----------------------------------------------------
                    # 6. Calcul du temps à partir des positions des moteurs
                    # -----------------------------------------------------
                    scanned = sample_processor.scanned_motors

                    # Position supposée constante sur les frames binnées :
                    # récupération de la position de la première frame réelle
                    # du groupe, et non de l'indice du bin.
                    position = {
                        motor: (
                            value[frame_group[0]]
                            if frame_binning == 1
                            else value[frame_group][0]
                        )
                        for motor, value in scanned.items()
                    }

                    # Vecteur [z, y] (moteur Z, moteur Y), voir docstring de convert_position_to_time
                    position_tuple = np.array([
                        next(
                            (v for m, v in position.items() if "z" in m),
                            0.0,
                        ),
                        next(
                            (v for m, v in position.items() if "y" in m),
                            0.0,
                        ),
                    ])

                    print(
                        f"position tuple for frame group {frame_group}: "
                        f"{position_tuple}"
                    )

                    time = convert_position_to_time(
                        position_tuple,
                        position0=position0,
                        
                    )

                    # -----------------------------------------------------
                    # 7. Extraction du PDF
                    # -----------------------------------------------------
                    r, G = extract_xpdf(
                        sample_processor,
                        ref_processor=ref_processor,
                        composition=composition,
                        rmin=rmin,
                        rmax=rmax,
                        rstep=rstep,
                        interactive=False,
                        plot=False,
                        bgscale=bgscale,
                        qmin=qmin,
                        qmax=qmax,
                        qmaxinst=qmaxinst,
                        rpoly=rpoly,
                       # Lorch=lorch,
                    )

                    # Stockage des résultats
                    res[str(frame_group)]["r"] = r
                    res[str(frame_group)]["G"] = G
                    res[str(frame_group)]["time"] = time

                    # -----------------------------------------------------
                    # 8. Tracé de G(r) décalé en fonction du temps
                    # -----------------------------------------------------
                    
                    progress_bar_gr_lostflow.progress(
                        (bin_idx + 1) / nb_bins,
                        text=f"Calcul de la PDF {bin_idx + 1}/{nb_bins}",
                    )

                # ---------------------------------------------------------
                # 9. Finalisation et stockage dans la session Streamlit
                # ---------------------------------------------------------
                progress_bar_gr_lostflow.empty()

                #st.session_state["fig_gr"] = fig
                st.session_state["res"] = res




#################################################################" STOP FLOW ########################################"




            
            elif option == "stop_flow":

                # ---------------------------------------------------------
                # 1. Initialisation des données et de la référence
                # ---------------------------------------------------------
                if entry_sample is None:
                    eiger = ID11Data(sample_file)
                    entry_sample = eiger.entries[0]

                nb_images = extract_number_of_images(
                    sample_file, entry_sample, poni_file
                )

                pdf_files = []

                # Appariement des frames échantillon / référence
                ref_frame_map = {}

                if ref_file is not None:
                    matches, unmatched = match_frames_by_motor_positions(
                        sample_file,
                        ref_file,
                        tol=position_tol,
                        entry_sample=entry_sample,
                        entry_ref=entry_ref,
                    )

                    ref_frame_map = {
                        m["sample_frame"]: m["ref_frame"]
                        for m in matches
                    }

                    if unmatched:
                        missing = [u["sample_frame"] for u in unmatched]
                        st.warning(
                            f"Aucune image de référence trouvée à "
                            f"{position_tol * 1000:.0f} µm près pour les "
                            f"frames échantillon {missing} ; indice identique "
                            f"utilisé en secours pour celles-ci."
                        )

                # ---------------------------------------------------------
                # 2. Définition du pas et calcul du nombre de groupes
                # ---------------------------------------------------------
                if frame_binning < 1:
                    raise ValueError("frame_binning must be >= 1.")

                if frame_step is None:
                    # Groupes consécutifs
                    frame_step_eff = frame_binning
                else:
                    # Espacement entre les débuts des groupes
                    if frame_step < frame_binning:
                        raise ValueError(
                            f"frame_step={frame_step} must be >= "
                            f"frame_binning={frame_binning}."
                        )

                    frame_step_eff = frame_step

                if nb_images < frame_binning:
                    nb_bins = 0
                else:
                    nb_bins = (
                        (nb_images - frame_binning) // frame_step_eff + 1
                    )

                if nb_bins == 0:
                    raise ValueError(
                        f"frame_binning={frame_binning} is larger than "
                        f"the number of images ({nb_images})."
                    )

                # Vérification des images restantes
                n_images_used = (
                    (nb_bins - 1) * frame_step_eff + frame_binning
                )

                if n_images_used < nb_images:
                    print(
                        f"Warning: {nb_images - n_images_used} trailing "
                        f"image(s) will not be used."
                    )

                if frame_step_eff > frame_binning:
                    print(
                        f"Info: {frame_step_eff - frame_binning} image(s) "
                        f"will be skipped between consecutive groups."
                    )

                st.session_state["nb_bins"] = nb_bins

                # ---------------------------------------------------------
                # 3. Initialisation du processeur et de la figure
                # ---------------------------------------------------------
                res = {}

                sample_processor = XRDProcessor(
                    sample_file,
                    frame=0,
                    poni_file=poni_file,
                    mask=mask_path,
                    dark=dark,
                    flat=flat,
                    spline=spline,
                    polarization_factor=polarization,
                    entry=entry_sample,
                )

                st.session_state["samplename"] = sample_processor.samplename

                
                progress_bar_gr_stopflow = st.progress(
                    0, text="Calcul du PDF en cours..."
                )

                # ---------------------------------------------------------
                # 4. Calcul du PDF pour chaque groupe
                # ---------------------------------------------------------
                for bin_idx in range(nb_bins):

                    
                    # Début du groupe courant
                    base = bin_idx * frame_step_eff

                    frame_group = [
                        base + k for k in range(frame_binning)
                    ]

                    frame_arg = (
                        frame_group[0]
                        if len(frame_group) == 1
                        else frame_group
                    )

                    res[str(frame_group)] = {}

                    print(f"Processing frame group: {frame_group}")

                    sample_processor = XRDProcessor(
                        sample_file,
                        frame=frame_arg,
                        poni_file=poni_file,
                        mask=mask_path,
                        dark=dark,
                        flat=flat,
                        spline=spline,
                        polarization_factor=polarization,
                        entry=entry_sample,
                    )

                    path_sf = Path(sample_file)

                    # Extraction du préfixe du sample
                    sample_name = path_sf.stem.split("_sf")[0]

                    # Reconstruction du chemin
                    filez2 = (
                        path_sf.parent.parent
                        / f"{sample_name}_scan_z2"
                        / f"{sample_name}_scan_z2.h5"
)
                    processor_scanz2 = XRDProcessor(filez2, frame=0, poni_file=poni_file,
                                            mask=mask_path, polarization_factor=polarization, entry=entry_sample)

                    # -----------------------------------------------------
                    # 5. Chargement de la référence correspondante
                    # -----------------------------------------------------
                    if ref_file is not None:
                        frame_group_ref = [
                            ref_frame_map.get(f, f)
                            for f in frame_group
                        ]

                        frame_arg_ref = (
                            frame_group_ref[0]
                            if len(frame_group_ref) == 1
                            else frame_group_ref
                        )

                        ref_processor = XRDProcessor(
                            ref_file,
                            frame=frame_arg_ref,
                            poni_file=poni_file,
                            mask=mask_path,
                            dark=dark,
                            flat=flat,
                            spline=spline,
                            polarization_factor=polarization,
                            entry=entry_ref,
                        )
                    else:
                        ref_processor = None

                    # -----------------------------------------------------
                    # 6. Calcul des temps de réaction
                    # -----------------------------------------------------
                    reaction_time = compute_reaction_time_stop_flow(
                        sample_processor, processor_scanz2
                    )

                    # Temps associé au groupe : celui de sa première frame
                    first_frame = frame_group[0]

                    if reaction_time is not None:
                        time = reaction_time[first_frame]
                    else:
                        time = None

                    # Conservation des temps associés aux frames du groupe
                    timedict = {}

                    if reaction_time is not None:
                        for frame in frame_group:
                            timedict[str(frame)] = reaction_time[frame]

                    res[str(frame_group)]["reaction_time"] = timedict

                    # -----------------------------------------------------
                    # 7. Extraction du PDF
                    # -----------------------------------------------------
                    r, G = extract_xpdf(
                        sample_processor,
                        ref_processor=ref_processor,
                        composition=composition,
                        rmin=rmin,
                        rmax=rmax,
                        rstep=rstep,
                        interactive=False,
                        plot=False,
                        bgscale=bgscale,
                        qmin=qmin,
                        qmax=qmax,
                        qmaxinst=qmaxinst,
                        rpoly=rpoly,
                       # Lorch=lorch,
                    )

                    res[str(frame_group)]["r"] = r
                    res[str(frame_group)]["G"] = G
                    res[str(frame_group)]["time"] = time

                    print("PDF computed for reaction time:", time)

                    # -----------------------------------------------------
                    # 8. Tracé de G(r)
                    # -----------------------------------------------------
                    

                    progress_bar_gr_stopflow.progress(
                        (bin_idx + 1) / nb_bins,
                        text=f"Calcul du PDF {bin_idx + 1}/{nb_bins}",
                    )

                # ---------------------------------------------------------
                # 9. Finalisation
                # ---------------------------------------------------------
                progress_bar_gr_stopflow.empty()

                #st.session_state["fig_gr"] = fig
                st.session_state["res"] = res


    # =========================================================
    # Reconstruction de la figure à partir des données stockées
    # =========================================================

    res = st.session_state.get("res")

    if res:

        # Option d'affichage
        apply_offset = st.checkbox(
            "Apply offset",
            value=True,
            key="apply_offset",
            help="Apply a vertical offset between successive PDF curves."
        )

        fig_gr_display = go.Figure()

        # Palette de couleurs
        nb_curves = len(res)

        palette = px.colors.sample_colorscale(
            "Bluered",
            [i / max(nb_curves - 1, 1) for i in range(nb_curves)],
        )

        # Titre
        option = cfg.option
        sample_name = st.session_state.get("samplename", "Unknown")

        fig_gr_display.update_layout(
            title=f"{option.replace('_', ' ').title()} - sample: {sample_name}",
            xaxis_title="r (Å)",
            yaxis_title="G(r)",
        )

        n = 0

        # Reconstruction des courbes
        for i, (frame_key, data) in enumerate(res.items()):

            r = data["r"]
            G = data["G"]
            time = data["time"]

            # Nom de la courbe
            if time is not None:
                curve_name = f"time={time:.2f}s"
            else:
                curve_name = f"frames={frame_key}"

            # Tracé
            fig_gr_display.add_trace(
                go.Scatter(
                    x=r,
                    y=G / G.max() + (n if apply_offset else 0),
                    mode="lines",
                    name=curve_name,
                    line=dict(
                        color=palette[i],
                        width=2,
                    ),
                )
            )

            if apply_offset:
                n += 0.02

        # Affichage dans Streamlit
        st.plotly_chart(
            fig_gr_display,
            use_container_width=True
        )

        # Ouverture dans une fenêtre séparée
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_gr"):

            fig_title = (
                f"G(r):{sample_name} "
                f"{option}"
            )

            open_fig_in_browser(
                fig_gr_display,
                tab_title=fig_title,
                filename=f"PDFs_{sample_name}.html"
            )
