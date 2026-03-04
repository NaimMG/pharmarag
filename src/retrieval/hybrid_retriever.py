"""
PharmaRAG — Hybrid Retriever (Dense + BM25)
Combine la recherche vectorielle BioBERT avec BM25
pour un retrieval plus précis sur les noms de médicaments
et termes médicaux exacts.

Dense (BioBERT) → comprend le sens sémantique
BM25            → retrouve les termes exacts
Ensemble        → meilleur des deux mondes
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger

# ── Configuration ───────────────────────────────────────
CHROMA_HOST      = "localhost"
CHROMA_PORT      = 8001
COLLECTION_NAME  = "pharmavigilance"
EMBEDDING_MODEL  = "dmis-lab/biobert-base-cased-v1.2"
PROCESSED_DIR    = Path("data/processed")

# Poids de la combinaison Dense + BM25
# 0.6 dense + 0.4 BM25 = privilégie la sémantique
DENSE_WEIGHT     = 0.6
BM25_WEIGHT      = 0.4


class HybridRetriever:
    """
    Retriever hybride combinant BioBERT (dense)
    et BM25 (sparse) pour la pharmacovigilance.

    Pourquoi hybride ?
    - Dense seul : rate parfois les termes médicaux exacts
    - BM25 seul  : ne comprend pas le contexte sémantique
    - Hybride    : précision + compréhension = meilleur recall
    """

    def __init__(self):
        logger.info("Initialisation HybridRetriever...")

        # ── BioBERT (Dense) ──────────────────────────
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ── ChromaDB ─────────────────────────────────
        self.client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # ── BM25 (Sparse) ────────────────────────────
        self.chunks, self.bm25 = self._build_bm25_index()

        logger.info(
            f"HybridRetriever prêt ✅\n"
            f"  Dense  : {self.collection.count()} vecteurs ChromaDB\n"
            f"  Sparse : {len(self.chunks)} documents BM25"
        )

    def _build_bm25_index(self):
        """
        Construit l'index BM25 depuis les chunks preprocessés.
        BM25 travaille sur des tokens (mots) — on tokenise
        simplement en splitant sur les espaces.
        """
        files = sorted(PROCESSED_DIR.glob("chunks_*.json"))
        if not files:
            raise FileNotFoundError("Aucun fichier chunks trouvé")

        latest = files[-1]
        logger.info(f"Construction index BM25 depuis : {latest.name}")

        with open(latest, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Tokenisation simple pour BM25
        tokenized = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized)
        logger.info(f"Index BM25 construit — {len(chunks)} documents")

        return chunks, bm25

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        """
        Recherche vectorielle dense avec BioBERT + ChromaDB.
        Retourne les chunks avec leurs scores de similarité.
        """
        query_vector = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings = [query_vector],
            n_results        = top_k * 2,  # On prend plus pour le merge
            include          = ["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            similarity = 1 - results["distances"][0][i]
            metadata   = results["metadatas"][0][i]
            chunks.append({
                "text":       doc,
                "source":     metadata.get("source", ""),
                "drug":       metadata.get("drug", ""),
                "report_id":  metadata.get("report_id", ""),
                "pmid":       metadata.get("pmid", ""),
                "dense_score": similarity,
                "bm25_score":  0.0,
                "hybrid_score": 0.0,
            })

        return chunks

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        Recherche BM25 (sparse) sur les chunks preprocessés.
        Excellente pour les termes médicaux exacts.
        """
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)

        # Récupérer les top_k*2 meilleurs scores
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k * 2]

        # Normaliser les scores BM25 entre 0 et 1
        max_score = max(scores[i] for i in top_indices) if top_indices else 1
        max_score = max_score if max_score > 0 else 1

        chunks = []
        for idx in top_indices:
            chunk    = self.chunks[idx]
            metadata = chunk.get("metadata", {})
            chunks.append({
                "text":        chunk["text"],
                "source":      metadata.get("source", ""),
                "drug":        metadata.get("drug", ""),
                "report_id":   metadata.get("report_id", ""),
                "pmid":        metadata.get("pmid", ""),
                "dense_score": 0.0,
                "bm25_score":  scores[idx] / max_score,
                "hybrid_score": 0.0,
            })

        return chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieval hybride principal.

        Algorithme :
        1. Dense search  → top_k*2 résultats avec scores
        2. BM25 search   → top_k*2 résultats avec scores
        3. Fusion        → score hybride = Dense*0.6 + BM25*0.4
        4. Déduplication → par texte exact
        5. Tri           → par score hybride décroissant
        6. Retourne top_k résultats
        """
        logger.info(f"Hybrid search : '{query[:60]}...'")

        # 1 + 2 : Recherches
        dense_results = self._dense_search(query, top_k)
        bm25_results  = self._bm25_search(query, top_k)

        # 3 : Fusion avec déduplication par texte
        merged = {}

        for chunk in dense_results:
            key = chunk["text"][:100]  # Clé de dédup
            if key not in merged:
                merged[key] = chunk.copy()
            merged[key]["dense_score"] = chunk["dense_score"]

        for chunk in bm25_results:
            key = chunk["text"][:100]
            if key not in merged:
                merged[key] = chunk.copy()
            merged[key]["bm25_score"] = chunk["bm25_score"]

        # 4 : Calcul du score hybride
        for key in merged:
            merged[key]["hybrid_score"] = round(
                DENSE_WEIGHT * merged[key]["dense_score"] +
                BM25_WEIGHT  * merged[key]["bm25_score"],
                4,
            )

        # 5 : Tri par score hybride
        results = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )[:top_k]

        logger.info(
            f"  Dense : {len(dense_results)} | "
            f"BM25 : {len(bm25_results)} | "
            f"Fusionnés : {len(results)} | "
            f"Score max : {results[0]['hybrid_score']:.3f}"
        )

        return results


if __name__ == "__main__":
    """Test comparatif Dense vs Hybride."""

    retriever = HybridRetriever()

    queries = [
        "ibuprofen gastrointestinal bleeding hospitalization",
        "aspirin fatal outcome death",
        "metformin lactic acidosis serious",
    ]

    for query in queries:
        print(f"\n{'='*55}")
        print(f"❓ {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['source'].upper()}] {r['drug']:12s} | "
                f"hybrid={r['hybrid_score']:.3f} | "
                f"dense={r['dense_score']:.3f} | "
                f"bm25={r['bm25_score']:.3f}"
            )
            print(f"     {r['text'][:80]}...")