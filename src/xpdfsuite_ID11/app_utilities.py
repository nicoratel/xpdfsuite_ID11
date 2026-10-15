import json
import glob
import os
import tempfile
import webbrowser
from pathlib import Path
import streamlit as st


CONFIG = Path.home() / ".id11_app.json"

def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except Exception:
        return {}

def save_config(**kw):
    cfg = load_config()
    cfg.update(kw)
    CONFIG.write_text(json.dumps(cfg))

@st.cache_data(ttl=60)
def list_files(root, pattern="*.h5"):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

# --- Dossier de recherche -------------------------------------------------
if "data_root" not in st.session_state:
    st.session_state["data_root"] = (
        load_config().get("data_root")
        or os.environ.get("ID11_DATA_ROOT")
        or str(Path.home())
    )

def _list_subdirs(path):
    """Liste les sous-dossiers accessibles (ignore les erreurs de permission)."""
    try:
        return sorted(
            d.name for d in Path(path).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return []

def _browse_go_parent():
    current = Path(st.session_state["browse_path"])
    st.session_state["browse_path"] = str(current.parent)

def _browse_enter_subdir():
    selected = st.session_state.get("browse_subdirs")
    if selected:
        current = Path(st.session_state["browse_path"])
        st.session_state["browse_path"] = str(current / selected)

def _browse_choose():
    # Exécuté par le callback on_click, donc AVANT que le widget
    # text_input(key="data_root") ne soit réinstancié : la modification
    # de st.session_state["data_root"] est alors autorisée.
    st.session_state["data_root"] = st.session_state["browse_path"]

def folder_browser():
    """Navigateur de dossiers natif Streamlit (ne dépend d'aucun binaire
    externe type zenity/tkinter, utilisable sur un serveur distant sans
    affichage graphique)."""
    if "browse_path" not in st.session_state:
        st.session_state["browse_path"] = st.session_state.get("data_root") or str(Path.home())

    current = Path(st.session_state["browse_path"])
    st.caption(f"📁 `{current}`")

    col_up, col_choose = st.columns(2)
    col_up.button("⬆️ Parent", key="browse_up", width='stretch', on_click=_browse_go_parent)
    col_choose.button("✅ Choisir", key="browse_choose", width='stretch', on_click=_browse_choose)

    subdirs = _list_subdirs(current)
    if subdirs:
        st.radio(
            "Sous-dossiers", subdirs, key="browse_subdirs",
            label_visibility="collapsed",
        )
        st.button("➡️ Entrer dans le dossier", key="browse_enter", on_click=_browse_enter_subdir)
    else:
        st.caption("Aucun sous-dossier")



# --- Sélecteur de fichier + champ d'affichage -------------------------------
def file_picker(label, key, files, root):
    def _on_select():
        st.session_state[f"{key}_path"] = st.session_state[f"{key}_select"] or ""

    st.selectbox(
        label, files, index=None, key=f"{key}_select",
        on_change=_on_select,
        format_func=lambda p: os.path.relpath(p, root),
        placeholder="Choisir un fichier...",
    )
    path = st.text_input("Fichier sélectionné", key=f"{key}_path",disabled=True)
    if path and not os.path.isfile(path):
        st.error("Fichier introuvable")
        return None
    return path or None


def open_fig_in_browser(fig, tab_title: str, filename: str):
    """Ouvre une figure Plotly dans un onglet nommé (au lieu de 127.0.0.1:port)."""
    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,   # "cdn" pour un fichier plus léger (nécessite internet)
        config={"responsive": True},
    )
    html = html.replace("<head>", f"<head><title>{tab_title}</title>", 1)
    path = Path(tempfile.gettempdir()) / filename
    path.write_text(html, encoding="utf-8")
    webbrowser.open(path.as_uri())