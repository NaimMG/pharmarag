"""
PharmaRAG — Streamlit Frontend
Interface chat pour interroger la base
de connaissances pharmacovigilance.
"""

import requests
import streamlit as st

# ── Configuration ───────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "PharmaRAG",
    page_icon  = "💊",
    layout     = "wide",
)

# ── CSS personnalisé ────────────────────────────────────
st.markdown("""
<style>
    /* Import de la police Syne */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&display=swap');

    /* Thème global Streamlit */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif !important;
    }

    /* Fond quadrillé de ton design original */
    .stApp {
        background-color: #0a0e1a;
        background-image:
          linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        color: #e2e8f0;
    }

    /* Style des cartes sources */
    .source-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .faers-card { border-left: 3px solid #f59e0b; }
    .pubmed-card { border-left: 3px solid #3b82f6; }

    /* Customisation de la Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1e2d45;
    }
    
    /* Boutons et inputs */
    .stTextInput>div>div>input {
        background-color: #1a2235;
        color: #e2e8f0;
        border: 1px solid #1e2d45;
    }
</style>
""", unsafe_allow_html=True)


# ── Fonctions utilitaires ───────────────────────────────

def check_api_health() -> dict | None:
    """Vérifie que l'API FastAPI est disponible."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        pass
    return None


def query_api(question: str) -> dict | None:
    """Envoie une question à l'API et retourne le résultat."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json    = {"question": question, "top_k": 5},
            timeout = 120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Erreur API : {resp.status_code} — {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle tourne sur le port 8000.")
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout — Llama3.2 prend trop de temps. Réessayez.")
    return None


# ── Header ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="color: #00d4aa; margin: 0;">💊 PharmaRAG</h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">
        Pharmacovigilance Intelligence — FDA FAERS + PubMed
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Statut du système")

    health = check_api_health()

    if health:
        st.success("API en ligne ✅")
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size: 2rem; color: #00d4aa; font-weight: bold;">
                {health['chroma_vectors']:,}
            </div>
            <div style="color: #64748b; font-size: 0.8rem;">vecteurs indexés</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        - **LLM** : `{health['llm_model']}`
        - **Collection** : `{health['collection']}`
        - **Embeddings** : `BioBERT`
        """)
    else:
        st.error("API hors ligne ❌")
        st.info("Lance l'API avec :\n```\npython -m uvicorn src.api.main:app --port 8000\n```")

    st.markdown("---")
    st.markdown("### 💡 Exemples de questions")

    example_questions = [
        "What are the serious adverse reactions for ibuprofen?",
        "Are there fatal outcomes with aspirin in FDA reports?",
        "What does literature say about metformin side effects?",
        "What hospitalizations are linked to amoxicillin?",
        "Compare adverse reactions of atorvastatin vs aspirin",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["selected_question"] = q

    st.markdown("---")
    st.markdown("""
    <div style="color: #64748b; font-size: 0.75rem;">
    ⚠️ Outil de recherche académique uniquement.<br>
    Ne constitue pas un avis médical.
    </div>
    """, unsafe_allow_html=True)


# ── Historique des conversations ─────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 Sources utilisées ({len(message['sources'])})"):
                for src in message["sources"]:
                    card_class = "faers-card" if src["source"] == "faers" else "pubmed-card"
                    label      = "🟡 FDA FAERS" if src["source"] == "faers" else "🔵 PubMed"
                    ref        = src.get("report_id") or src.get("pmid", "")
                    st.markdown(f"""
                    <div class="source-card {card_class}">
                        <strong>{label}</strong> — {src['drug'].upper()} | Ref: {ref}<br>
                        <small style="color: #94a3b8;">{src['text'][:200]}...</small>
                    </div>
                    """, unsafe_allow_html=True)


# ── Zone de saisie ──────────────────────────────────────

# Récupérer la question depuis les exemples sidebar
default_q = st.session_state.pop("selected_question", "")

question = st.chat_input(
    "Posez votre question sur la pharmacovigilance...",
)

# Utiliser la question du sidebar si cliquée
if default_q and not question:
    question = default_q

# ── Traitement de la question ───────────────────────────

if question:
    if not health:
        st.error("❌ L'API n'est pas disponible. Démarrez-la d'abord.")
        st.stop()

    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
    })

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans FDA FAERS + PubMed..."):
            result = query_api(question)

        if result:
            st.markdown(result["answer"])

            with st.expander(f"📚 Sources utilisées ({result['total_sources']})"):
                for src in result["sources"]:
                    card_class = "faers-card" if src["source"] == "faers" else "pubmed-card"
                    label      = "🟡 FDA FAERS" if src["source"] == "faers" else "🔵 PubMed"
                    ref        = src.get("report_id") or src.get("pmid", "")
                    st.markdown(f"""
                    <div class="source-card {card_class}">
                        <strong>{label}</strong> — {src['drug'].upper()} | Ref: {ref}<br>
                        <small style="color: #94a3b8;">{src['text'][:200]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })