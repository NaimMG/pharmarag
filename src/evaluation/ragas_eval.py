"""
PharmaRAG — Évaluation avec RAGAS
Mesure la qualité du système RAG sur 4 métriques :
- faithfulness      : la réponse est-elle fidèle aux sources ?
- answer_relevancy  : répond-elle vraiment à la question ?
- context_recall    : les bons documents sont-ils retrouvés ?
- context_precision : les documents retrouvés sont-ils pertinents ?

Usage : python src/evaluation/ragas_eval.py
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from loguru import logger

# ── Import de notre RAG Chain ───────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.generation.rag_chain import PharmaRAGChain

# ── Configuration ───────────────────────────────────────
EVAL_QUESTIONS_PATH = Path("data/eval/test_questions.json")
EVAL_RESULTS_PATH   = Path("data/eval")
EVAL_RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_test_questions() -> list[dict]:
    """Charge les questions de test depuis le fichier JSON."""
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"{len(questions)} questions de test chargées")
    return questions


def build_eval_dataset(
    rag_chain: PharmaRAGChain,
    questions: list[dict],
) -> dict:
    """
    Génère les réponses et contextes pour chaque question.
    Construit le dataset au format RAGAS.

    Format RAGAS attendu :
    - question      : la question posée
    - answer        : la réponse générée par le LLM
    - contexts      : liste des chunks récupérés (retrieval)
    - ground_truth  : la réponse de référence
    """
    eval_data = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for i, item in enumerate(questions):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Question {i+1}/{len(questions)}: {question[:60]}...")

        try:
            result   = rag_chain.query(question)
            answer   = result["answer"]
            contexts = [s["text"] for s in result["sources"]]

            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(ground_truth)

            logger.info(f"  ✅ Réponse générée ({len(contexts)} contextes)")

        except Exception as e:
            logger.error(f"  ❌ Erreur question {i+1}: {e}")
            continue

    return eval_data


def run_ragas_evaluation():
    """
    Pipeline complet d'évaluation RAGAS.
    """
    logger.info("=" * 55)
    logger.info("PharmaRAG — Évaluation RAGAS démarrée")
    logger.info("=" * 55)

    # ── 1. Initialiser la RAG Chain ─────────────────────
    logger.info("Initialisation RAG Chain...")
    rag_chain = PharmaRAGChain()

    # ── 2. Charger les questions ────────────────────────
    questions = load_test_questions()

    # ── 3. Générer les réponses ─────────────────────────
    logger.info("Génération des réponses pour toutes les questions...")
    eval_data = build_eval_dataset(rag_chain, questions)

    if not eval_data["question"]:
        logger.error("Aucune réponse générée — abandon")
        return

    # ── 4. Créer le dataset RAGAS ───────────────────────
    dataset = Dataset.from_dict(eval_data)
    logger.info(f"Dataset RAGAS créé : {len(dataset)} exemples")

    # ── 5. Configurer LLM + Embeddings pour RAGAS ───────
    # RAGAS a besoin d'un LLM et d'embeddings pour évaluer
    logger.info("Configuration LLM + Embeddings pour RAGAS...")

    llm = LangchainLLMWrapper(
        Ollama(model="llama3.2", temperature=0.0)
    )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name = "dmis-lab/biobert-base-cased-v1.2"
        )
    )

    # ── 6. Lancer l'évaluation RAGAS ────────────────────
    logger.info("Évaluation RAGAS en cours (patience ~5-10 min)...")

    results = evaluate(
        dataset    = dataset,
        metrics    = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm        = llm,
        embeddings = embeddings,
    )

    # ── 7. Afficher les résultats ────────────────────────
    scores = results.to_pandas()

    logger.info("\n" + "=" * 55)
    logger.info("✅ RÉSULTATS RAGAS")
    logger.info("=" * 55)

    metrics_summary = {
        "faithfulness":      float(np.nanmean(scores["faithfulness"])),
        "answer_relevancy":  float(np.nanmean(scores["answer_relevancy"])),
        "context_recall":    float(np.nanmean(scores["context_recall"])),
        "context_precision": float(np.nanmean(scores["context_precision"])) if not scores["context_precision"].isna().all() else None,
    }

    for metric, score in metrics_summary.items():
        if score != score:  # NaN check
            logger.info(f"  {metric:22s} : {'░' * 20} N/A (timeout)")
            continue
        bar   = "█" * int(score * 20)
        empty = "░" * (20 - int(score * 20))
        logger.info(f"  {metric:22s} : {bar}{empty} {score:.3f}")

    # ── 8. Sauvegarder les résultats ────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output    = EVAL_RESULTS_PATH / f"ragas_results_{timestamp}.json"

    def safe_convert(obj):
        """Convertit les types numpy en types Python natifs."""
        if isinstance(obj, float) and obj != obj:  # NaN
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    details = []
    for record in scores.to_dict(orient="records"):
        details.append({k: safe_convert(v) for k, v in record.items()})

    payload = {
        "evaluated_at":    timestamp,
        "total_questions": len(eval_data["question"]),
        "metrics":         metrics_summary,
        "details":         details,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"\nRésultats sauvegardés : {output}")
    logger.info("=" * 55)

    return metrics_summary


if __name__ == "__main__":
    run_ragas_evaluation()