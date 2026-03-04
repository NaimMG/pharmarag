"""
PharmaRAG — Script de lancement pipeline complet
Lance toutes les étapes dans l'ordre :
1. Ingestion FDA FAERS + PubMed
2. Preprocessing + Chunking
3. Embedding + Indexation ChromaDB

Usage : python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingestion.faers_ingestion import run_ingestion
from src.ingestion.pubmed_ingestion import run_pubmed_ingestion
from src.preprocessing.text_processor import run_preprocessing
from src.embeddings.embedder import run_embedding


def run_full_pipeline():
    logger.info("=" * 55)
    logger.info("PharmaRAG — Pipeline complet démarré")
    logger.info("=" * 55)

    # ── Étape 1 : Ingestion ──────────────────────────
    logger.info("\n📥 ÉTAPE 1/3 — Ingestion des données")
    logger.info("─" * 45)
    faers_summary  = run_ingestion(limit_per_drug=100)
    pubmed_total   = run_pubmed_ingestion(max_results=50)
    total_ingested = sum(s["count"] for s in faers_summary) + pubmed_total
    logger.info(f"✅ Ingestion terminée — {total_ingested} documents collectés")

    # ── Étape 2 : Preprocessing ──────────────────────
    logger.info("\n⚙️  ÉTAPE 2/3 — Preprocessing & Chunking")
    logger.info("─" * 45)
    stats = run_preprocessing()
    logger.info(f"✅ Preprocessing terminé — {stats['total_chunks']} chunks")

    # ── Étape 3 : Embedding ──────────────────────────
    logger.info("\n🧠 ÉTAPE 3/3 — Embedding & Indexation ChromaDB")
    logger.info("─" * 45)
    embed_stats = run_embedding()
    logger.info(f"✅ Indexation terminée — {embed_stats['total_vectors']} vecteurs")

    # ── Résumé final ─────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("🏁 PIPELINE TERMINÉ — Résumé")
    logger.info(f"   Documents collectés : {total_ingested}")
    logger.info(f"   Chunks générés      : {stats['total_chunks']}")
    logger.info(f"   Vecteurs indexés    : {embed_stats['total_vectors']}")
    logger.info("=" * 55)
    logger.info("\n➡️  Lance maintenant :")
    logger.info("   python -m uvicorn src.api.main:app --port 8000")
    logger.info("   streamlit run src/ui/app.py")


if __name__ == "__main__":
    run_full_pipeline()