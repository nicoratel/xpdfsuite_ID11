"""Tab 3: PDF extraction (same workflow as the monitor's « Extract xPDF » tab, for one frame / mean)."""

from __future__ import annotations

import json
import traceback
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from xpdfsuite_ID11 import XRDProcessor
from xpdfsuite_ID11.app_utilities import open_fig_in_browser
from xpdfsuite_ID11.pdf_extraction import auto_bgscale, compute_xPDF

from .configure_h5 import H5Config


def _make_processor(file, frame, entry, cfg: H5Config) -> XRDProcessor:
    return XRDProcessor(
        file, frame=frame, poni_file=cfg.poni_file, mask=cfg.mask_file,
        dark=cfg.dark_file, flat=cfg.flat_file, spline=cfg.spline_file,
        polarization_factor=cfg.polarization, entry=entry,
    )


def _integration_key(cfg: H5Config, sample_file, ref_file) -> tuple:
    """Identify the inputs that require re-running the (costly) azimuthal integration."""
    return (
        str(sample_file), cfg.frame_sample, cfg.entry_sample,
        str(ref_file), cfg.frame_ref, cfg.entry_ref,
        str(cfg.poni_file), str(cfg.mask_file), str(cfg.dark_file),
        str(cfg.flat_file), str(cfg.spline_file),
        cfg.polarization,
    )


def _ensure_integration(cfg: H5Config, sample_file, ref_file, force: bool = False) -> bool:
    """Run the azimuthal integration if inputs changed (or if forced).

    The cache key is stored ONLY on success, so a failure is retried next time.
    Returns True if integrated data are available.
    """
    key = _integration_key(cfg, sample_file, ref_file)
    ok = (
        st.session_state.get("_pdf_integration_key") == key
        and st.session_state.get("_pdf_q_sample") is not None
    )
    if ok and not force:
        return True

    st.session_state["_pdf_integration_key"] = None
    st.session_state["_pdf_integration_error"] = None
    st.session_state["_pdf_q_sample"] = None
    st.session_state["_pdf_I_sample"] = None
    st.session_state["_pdf_I_ref"] = None
    try:
        with st.spinner("Intégration azimutale en cours..."):
            sample_processor = _make_processor(sample_file, cfg.frame_sample, cfg.entry_sample, cfg)
            if not np.any(sample_processor.img):
                raise ValueError(
                    "The selected sample image is entirely zero. Check the entry/frame, and that "
                    "the H5 file's linked raw data files are next to it."
                )
            q_sample, I_sample = sample_processor.integrate(plot=False)

            I_ref = None
            if ref_file is not None:
                ref_processor = _make_processor(ref_file, cfg.frame_ref, cfg.entry_ref, cfg)
                q_ref, I_ref_raw = ref_processor.integrate(plot=False)
                I_ref = np.interp(q_sample, q_ref, I_ref_raw)

            st.session_state["samplename"] = sample_processor.samplename
            st.session_state["_pdf_q_sample"] = q_sample
            st.session_state["_pdf_I_sample"] = I_sample
            st.session_state["_pdf_I_ref"] = I_ref
            st.session_state["_pdf_integration_key"] = key  # only on success
        return True
    except Exception as exc:
        st.session_state["_pdf_integration_error"] = str(exc)
        return False


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

    # Small status recap so you can see what the tab actually receives
    st.caption(
        f"Échantillon : `{sample_file}` · Référence : `{ref_file if ref_file is not None else '— aucune —'}`"
    )

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
        qmin = st.slider(
            "qmin (Å⁻¹)", min_value=0.0, max_value=10.0,
            value=_default_qmin, step=0.01, key="qmin_slider",
        )
        qmax = st.slider(
            "qmax (Å⁻¹)", min_value=5.0, max_value=40.0,
            value=_default_qmax, step=0.01, key="qmax_slider",
        )

    with col_out3:
        st.markdown("**Paramètres pour la soustraction polynomiale**")
        qmaxinst = st.slider(
            "qmaxinst (Å⁻¹)", min_value=5.0, max_value=40.0,
            value=_default_qmaxinst, step=0.01, key="qmaxinst_slider",
            help="Maximum Q value for the instrument correction. If None, the maximum Q value will be used.",
        )
        rpoly = st.slider(
            "rpoly", min_value=0.0, max_value=3.0,
            value=_default_rpoly, step=0.01, key="rpoly_slider",
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
                    st.error("Un fichier de référence est requis pour la détection automatique.")
                elif not _ensure_integration(cfg, sample_file, ref_file):
                    st.error(f"Intégration indisponible : {st.session_state.get('_pdf_integration_error')}")
                else:
                    with st.spinner("Calcul du bgscale optimal..."):
                        try:
                            bg_opt, h_opt, r_found = auto_bgscale(
                                st.session_state["_pdf_q_sample"],
                                st.session_state["_pdf_I_sample"],
                                st.session_state["_pdf_I_ref"],
                                composition,
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
        bgscale = st.slider(
            "bgscale", min_value=0.0, max_value=2.5, step=0.01, key="bgscale_value",
            help="Scaling factor for the background subtraction. It should be kept < 1.5.",
        )
        st.session_state["bgscale"] = bgscale

    # ========== COMPUTE BUTTON ==========
    integration_key = _integration_key(cfg, sample_file, ref_file)
    extraction_params = dict(
        composition=composition, rmin=rmin, rmax=rmax, rstep=rstep,
        qmin=qmin, qmax=qmax, qmaxinst=qmaxinst, rpoly=rpoly,
        lorch=lorch, bgscale=bgscale,
    )
    params = {**extraction_params, "integration": integration_key}

    col_btn, col_auto, col_force = st.columns([1, 1.5, 1.5])
    with col_btn:
        compute_clicked = st.button("🚀 Calculer la PDF", type="primary", key="btn_compute_pdf")
    with col_auto:
        auto_recompute_enabled = st.checkbox(
            "Recalcul automatique (curseurs)", value=True, key="auto_recompute_pdf",
            help="Recalcule la PDF dès qu'un paramètre d'extraction change, une fois le premier calcul effectué.",
        )
    with col_force:
        force_reint = st.checkbox("Forcer la ré-intégration", value=False, key="force_reint")

    # Auto-recompute: only after a first computation, only if the integration is still valid
    # (same files/calibration), and only if an extraction parameter changed.
    last_params = st.session_state.get("_pdf_last_params")
    integration_ready = (
        st.session_state.get("_pdf_integration_key") == integration_key
        and st.session_state.get("_pdf_q_sample") is not None
    )
    params_changed = (
        last_params is not None
        and last_params.get("integration") == integration_key
        and {k: v for k, v in last_params.items() if k != "integration"} != extraction_params
    )
    auto_triggered = auto_recompute_enabled and integration_ready and params_changed

    if compute_clicked or auto_triggered:
        st.session_state["_pdf_calc_error"] = None
        if _ensure_integration(cfg, sample_file, ref_file, force=force_reint and compute_clicked):
            try:
                with st.spinner("Calcul de la PDF en cours..."):
                    r, G = compute_xPDF(
                        st.session_state["_pdf_q_sample"],
                        st.session_state["_pdf_I_sample"],
                        composition,
                        Iref=st.session_state.get("_pdf_I_ref"),
                        bgscale=bgscale,
                        qmin=qmin,
                        qmax=qmax,
                        qmaxinst=qmaxinst,
                        rmin=rmin,
                        rmax=rmax,
                        rstep=rstep,
                        rpoly=rpoly,
                        Lorch=lorch,
                        plot=False,
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
                st.session_state["_pdf_last_params"] = params
            except Exception:
                st.session_state["_pdf_calc_error"] = traceback.format_exc()

    # ========== STATUS / ERRORS ==========
    if st.session_state.get("_pdf_integration_error"):
        st.error(f"❌ Error integrating data: {st.session_state['_pdf_integration_error']}")
    if st.session_state.get("_pdf_calc_error"):
        st.error("❌ Error extracting PDF")
        with st.expander("Error details"):
            st.code(st.session_state["_pdf_calc_error"])

    if st.session_state.get("_pdf_q_sample") is not None and ref_file is None:
        st.info("ℹ️ Aucune référence sélectionnée : la PDF est calculée sans soustraction de fond.")

    # ========== DISPLAY ==========
    fig_gr_display = st.session_state.get("fig_gr")
    if fig_gr_display is not None:
        if st.session_state.get("_pdf_last_params") != params:
            st.warning("⚠️ Les paramètres ont changé depuis le dernier calcul : cliquez sur « Calculer la PDF ».")
        st.plotly_chart(fig_gr_display, width="stretch")
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