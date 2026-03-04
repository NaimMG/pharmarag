"""
PharmaRAG — Re-ranking avec Cross-Encoder
Améliore la qualité du retrieval en re-classant
les chunks récupérés par BioBERT.

Pipeline complet :
1. BioBERT    → récupère Top 20 chunks (recall large)
2. CrossEncoder → re-classe les 20 chunks (précision fine)
3. LLM        → reçoit uniquement les Top 5 meilleurs

Pourquoi Cross-Encoder ?
Un bi-encoder (BioBERT) encode question et chunk séparément.
Un cross-encoder les encode ENSEMBLE — il comprend mieux
la relation entre la question et le chunk, mais est plus lent.
On l'utilise donc uniquement pour re-classer, pas pour chercher.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger

# ── Configuration ───────────────────────────────────────
CHROMA_HOST      = "localhost"
CHROMA_PORT      = 8001
COLLECTION_NAME  = "pharmavigilance"
EMBEDDING_MODEL  = "dmis-lab/biobert-base-cased-v1.2"

# Cross-Encoder léger et performant pour le re-ranking médical
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_TOP_K  = 20   # On récupère large
RERANKING_TOP_K  = 5    # On garde les meilleurs


class PharmaReranker:
    """
    Module de re-ranking pour PharmaRAG.

    Étapes :
    1. Retrieval large  : BioBERT récupère top_retrieval chunks
    2. Re-ranking       : CrossEncoder re-classe par pertinence
    3. Résultat final   : top_reranking meilleurs chunks
    """

    def __init__(self):
        logger.info("Initialisation PharmaReranker...")

        # BioBERT pour le retrieval initial
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB
        self.client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # Cross-Encoder pour le re-ranking
        logger.info(f"Chargement Cross-Encoder : {RERANKER_MODEL}")
        self.cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info("Cross-Encoder chargé ✅")

        logger.info(
            f"PharmaReranker prêt ✅\n"
            f"  Retrieval : Top {RETRIEVAL_TOP_K} chunks (BioBERT)\n"
            f"  Reranking : Top {RERANKING_TOP_K} chunks (CrossEncoder)"
        )

    def retrieve_and_rerank(
        self,
        query:           str,
        top_retrieval:   int = RETRIEVAL_TOP_K,
        top_reranking:   int = RERANKING_TOP_K,
    ) -> list[dict]:
        """
        Pipeline complet retrieval + re-ranking.

        Args:
            query         : question en langage naturel
            top_retrieval : nombre de chunks récupérés initialement
            top_reranking : nombre de chunks après re-ranking

        Returns:
            Liste des meilleurs chunks re-classés avec scores
        """
        # ── Étape 1 : Retrieval large ────────────────────
        logger.info(f"Retrieval large : Top {top_retrieval} chunks...")
        query_vector = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings = [query_vector],
            n_results        = top_retrieval,
            include          = ["documents", "metadatas", "distances"],
        )

        # Formater les chunks récupérés
        candidates = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            candidates.append({
                "text":          doc,
                "source":        meta.get("source", ""),
                "drug":          meta.get("drug", ""),
                "report_id":     meta.get("report_id", ""),
                "pmid":          meta.get("pmid", ""),
                "dense_score":   round(1 - results["distances"][0][i], 4),
                "rerank_score":  0.0,
            })

        logger.info(f"  {len(candidates)} candidats récupérés")

        # ── Étape 2 : Re-ranking avec Cross-Encoder ──────
        logger.info("Re-ranking avec Cross-Encoder...")

        # Le Cross-Encoder prend des paires (question, chunk)
        pairs  = [[query, c["text"]] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Assigner les scores de re-ranking
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        # ── Étape 3 : Trier par score de re-ranking ──────
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )[:top_reranking]

        logger.info(
            f"Re-ranking terminé ✅\n"
            f"  Avant : {len(candidates)} chunks\n"
            f"  Après : {len(reranked)} chunks\n"
            f"  Meilleur score : {reranked[0]['rerank_score']:.4f}"
        )

        return reranked

    def compare_with_without_reranking(self, query: str) -> dict:
        """
        Compare les résultats avec et sans re-ranking.
        Utile pour démontrer l'amélioration en entretien.
        """
        # Sans re-ranking (dense seul, top 5)
        query_vector = self.embedder.encode(query).tolist()
        results      = self.collection.query(
            query_embeddings = [query_vector],
            n_results        = 5,
            include          = ["documents", "metadatas", "distances"],
        )

        without_reranking = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            without_reranking.append({
                "text":        doc[:100],
                "source":      meta.get("source", ""),
                "drug":        meta.get("drug", ""),
                "dense_score": round(1 - results["distances"][0][i], 4),
            })

        # Avec re-ranking
        with_reranking = self.retrieve_and_rerank(query, top_reranking=5)

        return {
            "query":              query,
            "without_reranking":  without_reranking,
            "with_reranking":     [
                {
                    "text":          r["text"][:100],
                    "source":        r["source"],
                    "drug":          r["drug"],
                    "dense_score":   r["dense_score"],
                    "rerank_score":  r["rerank_score"],
                }
                for r in with_reranking
            ],
        }


if __name__ == "__main__":
    """Test du re-ranking avec comparaison avant/après."""

    reranker = PharmaReranker()

    queries = [
        "ibuprofen serious adverse reactions hospitalization",
        "aspirin fatal outcome death FDA reports",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"❓ {query}")

        comparison = reranker.compare_with_without_reranking(query)

        print("\n📊 SANS re-ranking (Dense seul) :")
        for i, r in enumerate(comparison["without_reranking"], 1):
            print(f"  {i}. [{r['source'].upper()}] {r['drug']:12s} | dense={r['dense_score']:.4f} | {r['text'][:60]}...")

        print("\n✅ AVEC re-ranking (CrossEncoder) :")
        for i, r in enumerate(comparison["with_reranking"], 1):
            print(f"  {i}. [{r['source'].upper()}] {r['drug']:12s} | dense={r['dense_score']:.4f} | rerank={r['rerank_score']:.4f} | {r['text'][:60]}...")