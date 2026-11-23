"""Tab 2: Plot I(q) with reference subtraction."""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xpdfsuite_ID11 import XRDProcessor
from .configure_h5 import H5Config


def render(cfg: H5Config) -> None:
    """Render the I(q) plotting tab."""
    
    st.markdown("## 📈 Plot I(q) with Reference Subtraction")
    
    # Check required files
    if cfg.sample_file is None or cfg.poni_file is None:
        st.warning("⚠️ Please configure sample and PONI files in the Configuration tab.")
        return
    
    # Options for reference subtraction
    has_reference = cfg.ref_file is not None
    
    col1, col2 = st.columns(2)
    
    with col1:
        subtract_reference = st.checkbox(
            "Subtract reference data",
            value=has_reference,
            disabled=not has_reference,
            help="If checked, reference I(q) will be subtracted from sample I(q)"
        )
    
    with col2:
        if subtract_reference and has_reference:
            bgscale = st.slider(
                "Background scale factor",
                min_value=0.0,
                max_value=2.0,
                value=cfg.bgscale,
                step=0.01,
                help="Scale factor for reference subtraction"
            )
        else:
            bgscale = 1.0
    
    # Button to process and plot
    if st.button("📊 Process and Plot I(q)", key="btn_plot_iq", type="primary"):
        try:
            with st.spinner("Processing sample data..."):
                # Process sample
                processor_sample = XRDProcessor(
                    cfg.sample_file,
                    frame=cfg.frame_sample,
                    poni_file=cfg.poni_file,
                    mask=cfg.mask_file,
                    dark=cfg.dark_file,
                    flat=cfg.flat_file,
                    spline=cfg.spline_file,
                    polarization_factor=cfg.polarization,
                    entry=cfg.entry_sample,
                )
                if not np.any(processor_sample.img):
                    raise ValueError(
                        "The selected sample image is entirely zero. Check the entry/frame, "
                        "and that the H5 file's linked raw data files (e.g. scan0001/eiger_0000.h5) "
                        "are next to it."
                    )
                q_sample, I_sample = processor_sample.integrate()
            
            # Process reference if provided
            if has_reference and subtract_reference:
                with st.spinner("Processing reference data..."):
                    # Process reference
                    processor_ref = XRDProcessor(
                        cfg.ref_file,
                        frame=cfg.frame_ref,
                        poni_file=cfg.poni_file,
                        mask=cfg.mask_file,
                        dark=cfg.dark_file,
                        flat=cfg.flat_file,
                        spline=cfg.spline_file,
                        polarization_factor=cfg.polarization,
                        entry=cfg.entry_ref,
                    )
                    if not np.any(processor_ref.img):
                        raise ValueError("The selected reference image is entirely zero (check entry/frame and linked raw files).")
                    q_ref, I_ref = processor_ref.integrate()
                
                # Interpolate reference to sample q grid
                I_ref_interp = np.interp(q_sample, q_ref, I_ref)
            else:
                I_ref_interp = None
            
            # Apply q range filter if specified
            mask_q = (q_sample >= cfg.q_min) & (q_sample <= cfg.q_max)
            q_plot = q_sample[mask_q]
            I_plot = I_sample[mask_q]
            
            if I_ref_interp is not None:
                I_ref_plot = I_ref_interp[mask_q]
                I_corrected = I_plot - bgscale * I_ref_plot
            else:
                I_ref_plot = None
                I_corrected = I_plot
            
            # Create interactive plot with Plotly
            if I_ref_plot is not None:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("I(q) - Sample and Reference", "I(q) - After Reference Subtraction"),
                    shared_xaxes=True,
                    vertical_spacing=0.12
                )
                
                # Row 1: Sample and Reference
                fig.add_trace(
                    go.Scatter(
                        x=q_plot, y=I_plot,
                        mode='lines',
                        name='Sample I(q)',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Sample</b><br>Q: %{x:.4f}<br>I: %{y:.4e}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=q_plot, y=I_ref_plot * bgscale,
                        mode='lines',
                        name=f'Reference I(q) × {bgscale:.2f}',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='<b>Reference (scaled)</b><br>Q: %{x:.4f}<br>I: %{y:.4e}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Row 2: Corrected I(q)
                fig.add_trace(
                    go.Scatter(
                        x=q_plot, y=I_corrected,
                        mode='lines',
                        name='I(q) corrected',
                        line=dict(color='green', width=2),
                        hovertemplate='<b>Corrected I(q)</b><br>Q: %{x:.4f}<br>I: %{y:.4e}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_xaxes(title_text="Q (Å⁻¹)", row=2, col=1)
                fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
                fig.update_yaxes(title_text="Intensity (a.u.)", row=2, col=1)
                fig.update_layout(
                    height=700,
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(x=0.02, y=0.98)
                )
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=q_plot, y=I_plot,
                        mode='lines',
                        name='Sample I(q)',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Sample</b><br>Q: %{x:.4f}<br>I: %{y:.4e}<extra></extra>'
                    )
                )
                
                fig.update_xaxes(title_text="Q (Å⁻¹)")
                fig.update_yaxes(title_text="Intensity (a.u.)")
                fig.update_layout(
                    height=600,
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(x=0.02, y=0.98),
                    title_text="I(q) - Sample Only (No reference provided)"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store data in session state for use in other tabs
            st.session_state["q_plot"] = q_plot
            st.session_state["I_plot"] = I_plot
            st.session_state["I_ref_plot"] = I_ref_plot
            st.session_state["I_corrected"] = I_corrected
            st.session_state["bgscale_used"] = bgscale
            
            # Display statistics
            st.markdown("### 📊 Data Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Q range", f"{q_plot[0]:.3f} - {q_plot[-1]:.3f} Å⁻¹")
                st.metric("Sample I(q) max", f"{I_plot.max():.4e}")
            
            with col2:
                if I_ref_plot is not None:
                    st.metric("Reference I(q) max", f"{I_ref_plot.max():.4e}")
                    st.metric("Reference scale", f"{bgscale:.4f}")
                else:
                    st.markdown("**No reference**")
                    st.markdown("*Reference subtraction not available*")
            
            with col3:
                if I_ref_plot is not None:
                    st.metric("Corrected I(q) max", f"{I_corrected.max():.4e}")
                    st.metric("Correction ratio", f"{(I_plot.max() / I_corrected.max() if I_corrected.max() > 0 else 0):.2f}")
                else:
                    st.markdown("**Analysis**")
                    st.markdown(f"*Mean intensity: {I_plot.mean():.4e}*")
            
            st.success("✅ I(q) plot generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    # Display information about XRDProcessor
    with st.expander("ℹ️ About XRDProcessor integration"):
        st.markdown("""
        The integration process:
        1. Load the H5 file and extract 2D diffraction image
        2. Apply corrections (dark, flat, spline if provided)
        3. Apply mask if provided
        4. Perform 2D integration to get 1D I(q) curve using the PONI calibration
        5. Apply polarization factor correction
        
        **Parameters used:**
        - Polarization factor: Controls polarization correction in integration
        - Entry #: HDF5 entry number (for multi-dataset files)
        - Frame #: Frame index to integrate (or mean of all frames)
        """)
