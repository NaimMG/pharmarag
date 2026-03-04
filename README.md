# 💊 PharmaRAG — Pharmacovigilance Intelligence System

> Système RAG (Retrieval-Augmented Generation) appliqué à la pharmacovigilance.  
> Interrogez en langage naturel des milliers de rapports FDA FAERS et publications PubMed.  
> Stack 100% open-source.

---

## 🎯 Contexte & Problématique

Les équipes de pharmacovigilance dans les grandes entreprises pharmaceutiques
doivent analyser des milliers de rapports d'effets indésirables (FDA FAERS)
et de publications scientifiques (PubMed) pour détecter des signaux de sécurité.

**Ce projet répond à la question :**
> *"Comment permettre à un expert métier d'interroger en langage naturel
> une base de connaissances pharmacovigilance sans compétences techniques ?"*

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                     UTILISATEUR                          │
│   "Quels effets graves ont été signalés pour l'aspirine?"│
└──────────────────────┬──────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Streamlit Frontend  │  :8501
            │  Landing Page +      │
            │  Chat Interface      │
            └──────────┬──────────┘
                       │ REST API
            ┌──────────▼──────────┐
            │   FastAPI Backend    │  :8000
            │   + Monitoring SQLite│
            └──────┬──────┬───────┘
                   │      │
      ┌────────────▼─┐  ┌─▼────────────┐
      │  ChromaDB     │  │  Ollama       │
      │  Vector Store │  │  Llama3.2     │
      │  :8001        │  │  :11434       │
      └──────┬────────┘  └──────────────┘
             │
      ┌──────▼──────────────────────────────────────┐
      │              PIPELINE RAG                    │
      │  BioBERT → Hybrid Search → CrossEncoder      │
      │  (Dense + BM25)    (Re-ranking Top20→Top5)   │
      └──────┬───────────────────────────────────────┘
             │
      ┌──────▼──────────────────────────┐
      │          DATA SOURCES            │
      │  📋 FDA FAERS   📚 PubMed NCBI   │
      │  500 rapports   157 abstracts    │
      └──────────────────────────────────┘
```

---

## 🛠️ Stack Technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| LLM | Llama3.2 (Ollama) | Génération de réponses |
| Embeddings | BioBERT (HuggingFace) | Vectorisation texte médical |
| Sparse Search | BM25 (rank_bm25) | Recherche par mots-clés exacts |
| Re-ranking | CrossEncoder MiniLM | Re-classement Top20 → Top5 |
| Vector DB | ChromaDB | Stockage et recherche sémantique |
| RAG Framework | LangChain | Orchestration RAG |
| Mémoire | ConversationBufferWindowMemory | Contexte conversationnel (3 tours) |
| API | FastAPI | Backend REST |
| Monitoring | SQLite | Traçabilité requêtes + performance |
| UI | Streamlit (multi-pages) | Interface utilisateur dark theme |
| Infra | Docker Compose | Orchestration services |
| Data | FDA FAERS + PubMed | Sources pharmacovigilance |
| Évaluation | RAGAS | Métriques qualité RAG |
| Tests | pytest (12/12) | Couverture API complète |

---

## 📊 Données

| Source | Volume | Type |
|--------|--------|------|
| FDA FAERS | 500 rapports | Effets indésirables réels |
| PubMed NCBI | 157 abstracts | Publications scientifiques |
| **Total indexé** | **1 503 vecteurs** | ChromaDB (BioBERT 768d) |

Médicaments couverts : `ibuprofen` · `metformin` · `atorvastatin` · `amoxicillin` · `aspirin`

---

## 📈 Résultats RAGAS

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| `faithfulness` | 0.650 | Réponses fidèles aux sources |
| `context_recall` | 0.670 | Bons documents bien retrouvés |
| `context_precision` | 0.679 | Documents retrouvés pertinents |

> Évaluation sur 10 questions de pharmacovigilance avec Llama3.2 local.

---

## 🚀 Lancement rapide

### Prérequis
- Docker & Docker Compose
- Python 3.10+
- Ollama installé avec Llama3.2 : `ollama pull llama3.2`

### 1. Clone
```bash
git clone https://github.com/NaimMG/pharmarag.git
cd pharmarag
```

### 2. Environnement Python
```bash
python3 -m venv pharmarag-venv
source pharmarag-venv/bin/activate
pip install -r requirements.txt
```

### 3. Collecte des données + Pipeline complet
```bash
python scripts/run_pipeline.py
```

### 4. Lancer les services
```bash
bash scripts/start_services.sh

# Terminal 2 — API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 3 — Frontend
streamlit run src/ui/app.py
```

### 5. Accéder à l'interface

| Service | URL |
|---------|-----|
| 🎨 Streamlit UI | http://localhost:8501 |
| ⚡ FastAPI Docs | http://localhost:8000/docs |
| 📊 Monitoring | http://localhost:8000/monitoring |
| 🗄️ ChromaDB | http://localhost:8001 |

---

## 💬 Exemples de questions
```
❓ What are the most serious adverse reactions reported for ibuprofen?
❓ Are there any fatal outcomes associated with aspirin in FDA reports?
❓ What does the scientific literature say about metformin side effects?
❓ What hospitalizations are linked to amoxicillin?
❓ Compare the adverse reactions of atorvastatin and aspirin
```

---

## 🗂️ Structure du projet
```
pharmarag/
├── configs/
│   └── config.yaml           # Configuration centralisée
├── data/
│   ├── raw/                  # Données brutes FAERS + PubMed
│   ├── processed/            # Chunks nettoyés (1 569)
│   ├── embeddings/           # Vecteurs ChromaDB
│   ├── eval/                 # Questions test + résultats RAGAS
│   └── monitoring.db         # Base SQLite monitoring
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── scripts/
│   ├── run_pipeline.py       # Lance tout le pipeline en 1 commande
│   └── start_services.sh     # Démarre ChromaDB + vérifie Ollama
├── src/
│   ├── ingestion/            # FDA FAERS + PubMed collectors
│   ├── preprocessing/        # Chunking + nettoyage médical
│   ├── embeddings/           # BioBERT + indexation ChromaDB
│   ├── retrieval/            # Retriever + Hybrid + CrossEncoder
│   ├── generation/           # RAG Chain + Mémoire conversationnelle
│   ├── evaluation/           # RAGAS evaluation pipeline
│   ├── api/                  # FastAPI + Monitoring SQLite
│   └── ui/                   # Streamlit multi-pages dark theme
├── tests/                    # 12 tests pytest (100% passing)
├── docker-compose.yml
└── requirements.txt
```

---

## 🧪 Tests
```bash
pytest tests/test_api.py -v
# 12 passed in 2.74s ✅
```

---

## 📡 API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Statut API + ChromaDB + LLM |
| POST | `/query` | Question → Réponse + Sources |
| GET | `/stats` | Statistiques ChromaDB |
| GET | `/monitoring` | Métriques requêtes + performance |

---

## ⚠️ Avertissement

> Ce système est un **outil de recherche académique uniquement**.  
> Il ne constitue pas un avis médical ou réglementaire.  
> Les données FDA FAERS contiennent des rapports non validés cliniquement.

---

## 👤 Auteur

**Naim** — Étudiant ingénieur Data & IA  
Projet portfolio · Stack 100% open-source

[![GitHub](https://img.shields.io/badge/GitHub-NaimMG-black?logo=github)](https://github.com/NaimMG/pharmarag)