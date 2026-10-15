"""Onglet « Configure Experiment » : choix du type d'expérience, fichiers, paramètres de traitement."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import numpy as np
import streamlit as st

from xpdfsuite_ID11.app_utilities import file_picker, folder_browser, list_files, save_config


@dataclass
class ExperimentConfig:
    """Paramètres de l'expérience, produits par l'onglet Configure et lus par les autres."""

    sample_file: str | None
    ref_file: str | None
    poni_file: str | None
    mask_path: str | None
    polarization: float
    entry_sample: int | None
    entry_ref: int | None
    frame_binning: int
    frame_step: int | None
    
    option: str
    position_tol: float
    position0: tuple | None
    x_coude: float | None
    z_retour: float | None
    debit: float | None
    section: float | None
    section_tri: float | None
    long_tri: float | None


def render() -> ExperimentConfig:
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
            "Regroupement d'images consécutives (binning)",
            value=1,
            min_value=1,
            step=1,
            help="nb de frames consécutives moyennées dans une série")
        st.session_state["frame_binning"] = frame_binning if frame_binning is not None else 1
        
        frame_step = st.number_input(
                "Décalage entre les frames (None = pas de moyenne inter-séries)",
                value=None,
                step=1,
                help="décalage (en frames) (skip frames- permet de gagner du temps de traitement))")
        
        st.session_state["frame_step"] = frame_step if frame_step is not None else 1

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

    # Géométrie de la cellule : renseignée seulement pour l'expérience lost_flow
    position0 = x_coude = z_retour = debit = section = section_tri = long_tri = None

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

    return ExperimentConfig(
        sample_file=sample_file,
        ref_file=ref_file,
        poni_file=poni_file,
        mask_path=mask_path,
        polarization=polarization,
        entry_sample=entry_sample,
        entry_ref=entry_ref,
        frame_binning=frame_binning,
        frame_step=frame_step,
        option=option,
        position_tol=position_tol,
        position0=position0,
        x_coude=x_coude,
        z_retour=z_retour,
        debit=debit,
        section=section,
        section_tri=section_tri,
        long_tri=long_tri,
    )
