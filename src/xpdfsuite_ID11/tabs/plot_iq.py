"""Onglet « Plot raw data » : intégration des images et tracé de I(q)."""

from __future__ import annotations

import json
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from xpdfsuite_ID11 import XRDProcessor
from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.utilities_process import (
    compute_reaction_time_stop_flow,
    convert_position_to_time,
    extract_number_of_images,
    match_frames_by_motor_positions,
)

from .configure import ExperimentConfig


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
    position_tol = cfg.position_tol
    position0 = cfg.position0
    x_coude = cfg.x_coude
    z_retour = cfg.z_retour
    debit = cfg.debit
    section = cfg.section
    section_tri = cfg.section_tri
    long_tri = cfg.long_tri

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

                    # ---------------------------------------------------------
                    # 1. Définition du pas et calcul du nombre de groupes
                    # ---------------------------------------------------------
                    if frame_step is None:
                        # Groupes consécutifs, sans saut d'images
                        frame_step_eff = frame_binning
                    else:
                        # frame_step définit l'espacement entre deux groupes
                        if frame_step < frame_binning:
                            raise ValueError(
                                f"frame_step={frame_step} must be >= "
                                f"frame_binning={frame_binning}."
                            )
                        frame_step_eff = frame_step

                    if frame_binning < 1:
                        raise ValueError("frame_binning must be >= 1.")

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

                    # Indice de la dernière image utilisée (exclusif)
                    n_images_used = (
                        (nb_bins - 1) * frame_step_eff + frame_binning
                    )

                    if n_images_used < nb_images:
                        print(
                            f"Warning: {nb_images - n_images_used} trailing image(s) "
                            f"will not be used."
                        )

                    if frame_step_eff > frame_binning:
                        print(
                            f"Info: {frame_step_eff - frame_binning} image(s) "
                            f"will be skipped between consecutive groups."
                        )

                    st.session_state["nb_bins"] = nb_bins

                    # ---------------------------------------------------------
                    # 2. Initialisation du processeur et des couleurs
                    # ---------------------------------------------------------
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

                    palette = px.colors.sample_colorscale(
                        "Bluered",
                        [i / max(nb_bins - 1, 1) for i in range(nb_bins)],
                    )

                    # Ne stocke que les données brutes pour permettre de rejouer
                    # le tracé sans refaire les intégrations coûteuses.
                    bins_data = []

                    progress_bar_iq_lostflow = st.progress(
                        0,
                        text="Intégration des frames en cours...",
                    )

                    # ---------------------------------------------------------
                    # 3. Intégration de chaque groupe d'images
                    # ---------------------------------------------------------
                    for bin_idx in range(nb_bins):

                        couleur_courbe = palette[bin_idx % len(palette)]

                        # Début du groupe courant
                        base = bin_idx * frame_step_eff

                        # Images à intégrer dans ce groupe
                        frame_group = [
                            base + k for k in range(frame_binning)
                        ]

                        frame_arg = (
                            frame_group[0]
                            if len(frame_group) == 1
                            else frame_group
                        )

                        print(f"Integrating frame group: {frame_group}")

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

                        qsample, Isample = sample_processor.integrate()

                        # -----------------------------------------------------
                        # 4. Récupération des positions des moteurs
                        # -----------------------------------------------------
                        scanned = sample_processor.scanned_motors

                        # Position supposée constante sur les frames binnées :
                        # on récupère la position de la première frame du groupe.
                        position = {
                            motor: (
                                value[frame_group[0]]
                                if frame_binning == 1
                                else value[frame_group][0]
                            )
                            for motor, value in scanned.items()
                        }

                        # Vecteur [y, z]
                        position_tuple = np.array([
                            next(
                                (v for m, v in position.items() if "y" in m),
                                0.0,
                            ),
                            next(
                                (v for m, v in position.items() if "z" in m),
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
                            x_coude=x_coude,
                            z_retour=z_retour,
                            debit=debit,
                            section=section,
                            long_tri=long_tri,
                            section_tri=section_tri,
                            tol=0.1,
                        )

                        # -----------------------------------------------------
                        # 5. Intégration de la référence, si disponible
                        # -----------------------------------------------------
                        Iref = None

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

                            qref, Iref = ref_processor.integrate()

                        # -----------------------------------------------------
                        # 6. Stockage des données brutes
                        # -----------------------------------------------------
                        bins_data.append({
                            "time": time,
                            "qsample": qsample,
                            "Isample": Isample,
                            "Iref": Iref,
                            "color": couleur_courbe,
                        })

                        progress_bar_iq_lostflow.progress(
                            (bin_idx + 1) / nb_bins,
                            text=f"Intégration {bin_idx + 1}/{nb_bins}",
                        )

                    progress_bar_iq_lostflow.empty()

                    # ---------------------------------------------------------
                    # 7. Stockage dans la session Streamlit
                    # ---------------------------------------------------------
                    st.session_state["iq_raw_data"] = {
                        "option": "lost_flow",
                        "samplename": sample_processor.samplename,
                        "bins": bins_data,
                    }

                
                # ======================+ STOP FLOW =========================   
                
# ====================== STOP FLOW =========================
                elif option == "stop_flow":

                    # ---------------------------------------------------------
                    # 1. Définition du pas et calcul du nombre de groupes
                    # ---------------------------------------------------------
                    if frame_binning < 1:
                        raise ValueError("frame_binning must be >= 1.")

                    if frame_step is None:
                        # Groupes consécutifs, sans saut d'images
                        frame_step_eff = frame_binning
                    else:
                        # frame_step définit l'espacement entre les débuts des groupes
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

                    # Indice de la dernière image utilisée (exclusif)
                    n_images_used = (
                        (nb_bins - 1) * frame_step_eff + frame_binning
                    )

                    if n_images_used < nb_images:
                        print(
                            f"Warning: {nb_images - n_images_used} trailing image(s) "
                            f"will not be used."
                        )

                    if frame_step_eff > frame_binning:
                        print(
                            f"Info: {frame_step_eff - frame_binning} image(s) "
                            f"will be skipped between consecutive groups."
                        )

                    st.session_state["nb_bins"] = nb_bins

                    # ---------------------------------------------------------
                    # 2. Initialisation du processeur et des couleurs
                    # ---------------------------------------------------------
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

                    palette = px.colors.sample_colorscale(
                        "Bluered",
                        [i / max(nb_bins - 1, 1) for i in range(nb_bins)],
                    )

                    # Stockage des données brutes, sans construire la figure
                    bins_data = []

                    progress_bar_iq_stopflow = st.progress(
                        0,
                        text="Intégration des frames en cours...",
                    )

                    # ---------------------------------------------------------
                    # 3. Intégration de chaque groupe d'images
                    # ---------------------------------------------------------
                    for bin_idx in range(nb_bins):

                        couleur_courbe = palette[bin_idx % len(palette)]

                        # Début du groupe courant
                        base = bin_idx * frame_step_eff

                        # Images constituant le groupe
                        frame_group = [
                            base + k for k in range(frame_binning)
                        ]

                        frame_arg = (
                            frame_group[0]
                            if len(frame_group) == 1
                            else frame_group
                        )

                        # Première frame du groupe, utilisée pour associer le temps
                        time_arg = frame_group[0]

                        print(f"Integrating frame group: {frame_group}")

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

                        qsample, Isample = sample_processor.integrate()

                        # -----------------------------------------------------
                        # 4. Récupération du temps de réaction
                        # -----------------------------------------------------
                        reaction_time = compute_reaction_time_stop_flow(
                            sample_processor
                        )

                        
                        if reaction_time is not None:
                            print(f"Reaction time for frame group {frame_group}: {reaction_time}")
                            time_arg = frame_group[0]
                            print(f"Reaction time for frame group {frame_group}: {reaction_time}")
                            print(f"Time argument for frame group {frame_group}: {time_arg}")
                            time = reaction_time[time_arg]
                            
                        else:
                            time = None

                        


                        # -----------------------------------------------------
                        # 5. Intégration de la référence, si disponible
                        # -----------------------------------------------------
                        Iref = None

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

                            qref, Iref = ref_processor.integrate()

                        # -----------------------------------------------------
                        # 6. Stockage des données brutes
                        # -----------------------------------------------------
                        bins_data.append({
                            "time": time,
                            "qsample": qsample,
                            "Isample": Isample,
                            "Iref": Iref,
                            "color": couleur_courbe,
                        })

                        progress_bar_iq_stopflow.progress(
                            (bin_idx + 1) / nb_bins,
                            text=f"Intégration {bin_idx + 1}/{nb_bins}",
                        )

                    progress_bar_iq_stopflow.empty()

                    # ---------------------------------------------------------
                    # 7. Stockage dans la session Streamlit
                    # ---------------------------------------------------------
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
