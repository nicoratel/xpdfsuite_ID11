"""Détection des mauvaises références (bulle d'air) pour les onglets Streamlit (lost_flow)."""

from __future__ import annotations

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from xpdfsuite_ID11.utilities_process import (
    prepare_bad_reference_correction,
    print_bad_reference_report,
)


@st.cache_resource(show_spinner="Détection des mauvaises références...")
def _cached_correction(ref_file, entry_ref, mask_path, factor, mtime):
    # mtime fait partie de la clé de cache : un fichier modifié est retraité.
    return prepare_bad_reference_correction(
        ref_file, entry_ref=entry_ref, mask_file=mask_path, factor=factor)


def get_ref_correction(cfg):
    """Correction de refs pour la config courante, ou None (non lost_flow, pas de ref, désactivée)."""
    if cfg.option != "lost_flow" or cfg.ref_file is None or not cfg.bad_ref_factor:
        return None
    return _cached_correction(cfg.ref_file, cfg.entry_ref, cfg.mask_path,
                              float(cfg.bad_ref_factor), os.path.getmtime(cfg.ref_file))


def render_report(correction, ref_frame_map=None) -> None:
    """Affiche le tri des refs dans l'app (+ print terminal)."""
    if correction is None:
        return
    print_bad_reference_report(correction, ref_frame_map)
    intensities = correction["intensities"]
    bad = sorted(correction["bad"])
    n = len(intensities)
    level, factor = correction["ref_level"], correction["factor"]

    if not bad:
        st.success(f"Références : aucune mauvaise ref détectée sur {n} frames.")
    else:
        st.warning(f"Références : {len(bad)}/{n} mauvaise(s) ref(s) détectée(s) (frames {bad}), "
                   f"remplacée(s) par la moyenne des {n - len(bad)} bonnes refs.")

    with st.expander("Détail du tri des références"):
        idx = np.arange(n)
        is_bad = np.isin(idx, bad)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx[~is_bad], y=intensities[~is_bad], mode="markers",
                                 name="bonne ref", marker=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=idx[is_bad], y=intensities[is_bad], mode="markers",
                                 name="mauvaise ref", marker=dict(color="red", size=9)))
        fig.add_hline(y=factor * level, line_dash="dash",
                      annotation_text=f"seuil = {factor:.2f} x bonne ref")
        fig.update_layout(xaxis_title="frame de référence",
                          yaxis_title="intensité moyenne (pixels non masqués)")
        st.plotly_chart(fig, use_container_width=True)
