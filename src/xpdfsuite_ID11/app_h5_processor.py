"""
Application Streamlit pour le traitement de fichiers H5 avec intégration I(q) et extraction PDF.
Fonctionnalités basiques : traitement H5 -> I(q) avec soustraction de référence -> PDF extraction.
"""

import streamlit as st
import tempfile
from pathlib import Path

from tabs_h5 import configure_h5, plot_iq_h5, extract_pdf_h5


# Configure Streamlit page
st.set_page_config(
    page_title="H5 Processor - PDF Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("H5 Processor - PDF Analysis")

# Add CSS to style tab labels
st.markdown("""
    <style>
        button[data-baseweb="tab"] {
            font-size: 14px !important;
            padding: 10px 20px !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 14px;
        }
        [role="tabpanel"] p {
            font-size: 12px !important;
        }
        [role="tabpanel"] h2 {
            font-size: 16px !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        [role="tabpanel"] h3 {
            font-size: 14px !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.4rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Add stop button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Stop App", type="secondary"):
    st.success("👋 Thanks for using H5 Processor!")
    st.stop()

# Create three tabs
tab1, tab2, tab3 = st.tabs([
    "⚙️ Configure Files", 
    "📈 Plot I(q) with Reference Subtraction", 
    "🧮 Extract PDF"
])

with tab1:
    cfg = configure_h5.render()
    # Save configuration to session_state so other tabs can access it
    st.session_state["h5_config"] = cfg

with tab2:
    # Get configuration from session_state
    cfg = st.session_state.get("h5_config")
    if cfg is not None:
        plot_iq_h5.render(cfg)
    else:
        st.info("👈 Please configure files in the **Configure Files** tab first.")

with tab3:
    # Get configuration from session_state
    cfg = st.session_state.get("h5_config")
    if cfg is not None:
        extract_pdf_h5.render(cfg)
    else:
        st.info("👈 Please configure files in the **Configure Files** tab first.")
