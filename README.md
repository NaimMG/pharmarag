# 💊 PharmaRAG — Pharmacovigilance Intelligence System

> Système RAG (Retrieval-Augmented Generation) appliqué à la pharmacovigilance.  
> Interrogez en langage naturel des milliers de rapports FDA FAERS et publications PubMed.  
> Stack 100% open-source — 0€ d'infrastructure.

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
            └──────────┬──────────┘
                       │ REST API
            ┌──────────▼──────────┐
            │   FastAPI Backend    │  :8000
            └──────┬──────┬───────┘
                   │      │
      ┌────────────▼─┐  ┌─▼────────────┐
      │  ChromaDB     │  │  Ollama       │
      │  Vector Store │  │  Llama3.2     │
      │  :8001        │  │  :11434       │
      └──────┬────────┘  └──────────────┘
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
| Vector DB | ChromaDB | Stockage et recherche sémantique |
| RAG Framework | LangChain | Orchestration RAG |
| API | FastAPI | Backend REST |
| UI | Streamlit | Interface utilisateur |
| Infra | Docker Compose | Orchestration services |
| Data | FDA FAERS + PubMed | Sources pharmacovigilance |
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

### 3. Collecte des données
```bash
python src/ingestion/faers_ingestion.py
python src/ingestion/pubmed_ingestion.py
```

### 4. Preprocessing + Indexation
```bash
python src/preprocessing/text_processor.py
python src/embeddings/embedder.py
```

### 5. Lancer les services
```bash
# Terminal 1 — ChromaDB
docker run -d --name pharmarag-chromadb \
  -p 8001:8000 \
  -v $(pwd)/data/embeddings:/chroma/chroma \
  chromadb/chroma:0.5.3

# Terminal 2 — API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 3 — Frontend
streamlit run src/ui/app.py
```

### 6. Accéder à l'interface
| Service | URL |
|---------|-----|
| 🎨 Streamlit UI | http://localhost:8501 |
| ⚡ FastAPI Docs | http://localhost:8000/docs |
| 🗄️ ChromaDB | http://localhost:8001 |

---

## 💬 Exemples de questions
```
❓ What are the most serious adverse reactions reported for ibuprofen?

❓ Are there any fatal outcomes associated with aspirin in FDA reports?

❓ What does the scientific literature say about metformin side effects?

❓ What hospitalizations are linked to amoxicillin?

❓ Compare the adverse reactions of atorvastatin and aspirin based on FDA and PubMed data
```

---

## 🗂️ Structure du projet
```
pharmarag/
├── data/
│   ├── raw/              # Données brutes FAERS + PubMed
│   └── processed/        # Chunks nettoyés
├── src/
│   ├── ingestion/        # Collecte FDA FAERS + PubMed
│   ├── preprocessing/    # Chunking + nettoyage
│   ├── embeddings/       # BioBERT + indexation ChromaDB
│   ├── retrieval/        # Recherche sémantique
│   ├── generation/       # RAG Chain LangChain + Llama3.2
│   ├── api/              # FastAPI backend
│   └── ui/               # Streamlit frontend
├── tests/                # 12 tests pytest (100% passing)
├── docker/               # Dockerfiles
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

## ⚠️ Avertissement

> Ce système est un **outil de recherche académique uniquement**.  
> Il ne constitue pas un avis médical ou réglementaire.  
> Les données FDA FAERS contiennent des rapports non validés cliniquement.

---

## 👤 Auteur

**Naim** — Étudiant ingénieur Data & IA  
Projet portfolio · Stack 100% open-source · 0€ d'infrastructure

[![GitHub](https://img.shields.io/badge/GitHub-NaimMG-black?logo=github)](https://github.com/NaimMG/pharmarag)