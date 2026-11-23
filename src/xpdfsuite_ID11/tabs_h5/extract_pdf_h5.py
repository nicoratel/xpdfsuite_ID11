"""Tab 3: PDF extraction (same workflow as the monitor's « Extract xPDF » tab, for one frame / mean)."""

from __future__ import annotations

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from xpdfsuite_ID11 import extract_xpdf, XRDProcessor
from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.pdf_extraction import auto_bgscale

from .configure_h5 import H5Config


def _make_processor(file, frame, entry, cfg: H5Config) -> XRDProcessor:
    return XRDProcessor(
        file, frame=frame, poni_file=cfg.poni_file, mask=cfg.mask_file,
        dark=cfg.dark_file, flat=cfg.flat_file, spline=cfg.spline_file,
        polarization_factor=cfg.polarization, entry=entry,
    )


def render(cfg: H5Config) -> None:
    sample_file = cfg.sample_file
    ref_file = cfg.ref_file

    st.markdown("## 🧮 Extract PDF")
    # ========== DEFAULT VALUES ==========
    _default_bgscale = st.session_state.get("bgscale", 1.0)
    _default_qmin = 0.5
    _default_qmax = 24.5
    _default_qmaxinst = 24.5
    _default_rpoly = 0.9
    _default_lorch = True
    _default_composition = "Au"

    if sample_file is None or cfg.poni_file is None:
        st.warning("⚠️ Please select the sample H5 file and the PONI file in the Configuration tab.")
        return

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
        st.markdown("**Gamme de R souhaitée**")
        rmin = st.number_input("rmin (Å)", value=0.0, step=0.1)
        rmax = st.number_input("rmax (Å)", value=50.0, step=0.1)
        rstep = st.number_input("rstep (Å)", value=0.01, step=0.001)

    with col_out2:
        st.markdown("**Gamme de Q utilisée pour l'extraction**")
        qmin = st.number_input("qmin (Å⁻¹)", value=_default_qmin, step=0.01)
        qmax = st.number_input("qmax (Å⁻¹)", value=_default_qmax, step=0.01)

    with col_out3:
        st.markdown("**Paramètres pour la soustraction polynomiale**")
        qmaxinst = st.number_input(
            "qmaxinst (Å⁻¹)", value=_default_qmaxinst, step=0.01,
            help="Maximum Q value for the instrument correction. If None, the maximum Q value will be used.",
        )
        rpoly = st.number_input(
            "rpoly", value=_default_rpoly, step=0.01,
            help="Polynomial correction parameter. It should be kept < 1.5.",
        )
        lorch = st.checkbox(
            "Lorch", value=_default_lorch,
            help="Apply Lorch modification to the Fourier transform. It is recommended to keep this option checked.",
        )

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
                if ref_file is None:
                    st.error("Fichiers échantillon et référence requis pour la détection automatique.")
                else:
                    with st.spinner("Calcul du bgscale optimal..."):
                        sample_processor0 = _make_processor(sample_file, cfg.frame_sample, cfg.entry_sample, cfg)
                        ref_processor0 = _make_processor(ref_file, cfg.frame_ref, cfg.entry_ref, cfg)
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
        try:
            with st.spinner("Calcul de la PDF en cours..."):
                sample_processor = _make_processor(sample_file, cfg.frame_sample, cfg.entry_sample, cfg)
                if not np.any(sample_processor.img):
                    raise ValueError(
                        "The selected sample image is entirely zero. Check the entry/frame, and that "
                        "the H5 file's linked raw data files are next to it."
                    )

                ref_processor = None
                if ref_file is not None:
                    ref_processor = _make_processor(ref_file, cfg.frame_ref, cfg.entry_ref, cfg)

                st.session_state["samplename"] = sample_processor.samplename

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

            fig = go.Figure(
                layout=go.Layout(
                    title=f"PDF - sample: {st.session_state['samplename']}",
                    xaxis_title="r (Å)",
                    yaxis_title="G(r)",
                )
            )
            fig.add_trace(
                go.Scatter(x=r, y=G, mode="lines", name="G(r)", line=dict(color="blue", width=2))
            )

            st.session_state["fig_gr"] = fig
            st.session_state["res"] = {"frame": {"r": r, "G": G}}
        except Exception as exc:
            st.error(f"❌ Error extracting PDF: {exc}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    fig_gr_display = st.session_state.get("fig_gr")
    if fig_gr_display is not None:
        st.plotly_chart(fig_gr_display, width='stretch')
        if st.button("🪟 Ouvrir dans une fenêtre séparée", key="open_window_gr"):
            samplename = st.session_state.get("samplename", "Unknown")
            open_fig_in_browser(
                fig_gr_display,
                tab_title=f"G(r):{samplename}",
                filename=f"PDFs_{samplename}.html",
            )

    if st.session_state.get("res") is not None:
        json_res = json.dumps(
            st.session_state.get("res"), indent=4,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
        st.download_button(
            label="Download PDF data as JSON",
            data=json_res,
            file_name=f"pdf_data_sample_{st.session_state.get('samplename')}.json",
            mime="application/json",
            key="download_pdf_data",
        )
