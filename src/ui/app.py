"""
PharmaRAG — Présentation de l'Architecture
Point d'entrée de l'application Streamlit.
"""

import streamlit as st
import streamlit.components.v1 as components
import os

# Configuration de la page principale
st.set_page_config(
    page_title="PharmaRAG | Architecture",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemin absolu vers le fichier HTML pour éviter les erreurs de chemin (Docker/Local)
current_dir = os.path.dirname(os.path.abspath(__file__))
html_file_path = os.path.join(current_dir, "architecture.html")

# Lecture et injection du code HTML
try:
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    
    # Affichage du HTML (hauteur ajustée pour englober tout ton design)
    components.html(html_code, height=1400, scrolling=True)

except FileNotFoundError:
    st.error(f"Fichier de présentation introuvable : {html_file_path}")