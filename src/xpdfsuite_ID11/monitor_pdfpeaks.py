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

from tabs import configure, plot_iq, fit_iq, extract_pdf, gr_evolution

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
    cfg = configure.render()
with tabiq:
    plot_iq.render(cfg)
with tabfitiq:
    fit_iq.render()
with tab2:
    extract_pdf.render(cfg)
with tab3:
    gr_evolution.render()