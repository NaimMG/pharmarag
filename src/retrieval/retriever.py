"""
PharmaRAG — Retrieval Module
Logique de recherche sémantique dans ChromaDB.
Séparé de la RAG Chain pour plus de modularité.

Permet à terme d'implémenter facilement :
- Hybrid Search (Dense + BM25)
- Re-ranking avec cross-encoder
- Filtrage par métadonnées
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger


# ── Configuration ───────────────────────────────────────
CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8001
COLLECTION_NAME = "pharmavigilance"
EMBEDDING_MODEL = "dmis-lab/biobert-base-cased-v1.2"


class PharmaRetriever:
    """
    Module de retrieval sémantique pour PharmaRAG.

    Responsabilités :
    - Vectoriser les requêtes avec BioBERT
    - Rechercher les chunks pertinents dans ChromaDB
    - Filtrer par métadonnées (source, drug, date)
    - Dédupliquer les résultats
    """

    def __init__(self):
        logger.info("Initialisation PharmaRetriever...")

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        logger.info(f"Retriever prêt — {self.collection.count()} vecteurs ✅")

    def retrieve(
        self,
        query:      str,
        top_k:      int  = 5,
        source:     str  = None,
        drug:       str  = None,
    ) -> list[dict]:
        """
        Recherche sémantique principale.

        Args:
            query  : question en langage naturel
            top_k  : nombre de résultats à retourner
            source : filtrer par source ('faers' ou 'pubmed')
            drug   : filtrer par médicament ('ibuprofen', etc.)

        Returns:
            Liste de chunks avec texte + métadonnées + score
        """
        # Vectoriser la requête
        query_vector = self.embedder.encode(query).tolist()

        # Construire les filtres ChromaDB
        where = self._build_filters(source=source, drug=drug)

        # Recherche dans ChromaDB
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results":        top_k,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Formater les résultats
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # Convertir distance cosine en score de similarité
            similarity = 1 - distance

            chunks.append({
                "text":       doc,
                "source":     metadata.get("source", ""),
                "drug":       metadata.get("drug", ""),
                "report_id":  metadata.get("report_id", ""),
                "pmid":       metadata.get("pmid", ""),
                "serious":    metadata.get("serious", ""),
                "similarity": round(similarity, 4),
            })

        logger.info(
            f"Retrieval : '{query[:50]}...' "
            f"→ {len(chunks)} chunks "
            f"(score max: {chunks[0]['similarity']:.3f})"
        )

        return chunks

    def retrieve_by_drug(self, drug: str, top_k: int = 10) -> list[dict]:
        """
        Récupère tous les chunks pour un médicament donné.
        Utile pour générer des résumés par médicament.
        """
        return self.retrieve(
            query  = f"adverse reactions side effects {drug}",
            top_k  = top_k,
            drug   = drug,
        )

    def retrieve_faers_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Recherche uniquement dans les rapports FDA FAERS."""
        return self.retrieve(query=query, top_k=top_k, source="faers")

    def retrieve_pubmed_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Recherche uniquement dans les articles PubMed."""
        return self.retrieve(query=query, top_k=top_k, source="pubmed")

    def _build_filters(
        self,
        source: str = None,
        drug:   str = None,
    ) -> dict | None:
        """
        Construit les filtres ChromaDB.
        ChromaDB utilise une syntaxe de type MongoDB.
        """
        filters = []

        if source:
            filters.append({"source": {"$eq": source}})
        if drug:
            filters.append({"drug": {"$eq": drug}})

        if len(filters) == 0:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la collection."""
        total = self.collection.count()
        return {
            "total_vectors": total,
            "collection":    COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }


if __name__ == "__main__":
    """Test rapide du retriever."""
    retriever = PharmaRetriever()

    # Test 1 : recherche générale
    results = retriever.retrieve(
        "serious adverse reactions ibuprofen hospitalization",
        top_k=3,
    )
    print("\n🔍 Test recherche générale :")
    for r in results:
        print(f"  [{r['source'].upper()}] {r['drug']} | score: {r['similarity']} | {r['text'][:80]}...")

    # Test 2 : FAERS uniquement
    results = retriever.retrieve_faers_only("aspirin fatal outcome", top_k=3)
    print("\n🔍 Test FAERS uniquement :")
    for r in results:
        print(f"  [{r['source'].upper()}] {r['drug']} | score: {r['similarity']} | {r['text'][:80]}...")

    # Test 3 : par médicament
    results = retriever.retrieve_by_drug("metformin", top_k=3)
    print("\n🔍 Test par médicament (metformin) :")
    for r in results:
        print(f"  [{r['source'].upper()}] {r['drug']} | score: {r['similarity']} | {r['text'][:80]}...")