"""
PharmaRAG — Embeddings & Indexation ChromaDB
Charge les chunks preprocessés, les vectorise
avec BioBERT et les indexe dans ChromaDB.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

# ── Configuration ───────────────────────────────────────
PROCESSED_DIR    = Path("data/processed")
CHROMA_HOST      = "localhost"
CHROMA_PORT      = 8001
COLLECTION_NAME  = "pharmavigilance"
EMBEDDING_MODEL  = "dmis-lab/biobert-base-cased-v1.2"
BATCH_SIZE       = 32   # Nombre de chunks vectorisés en une fois


class PharmaEmbedder:
    """
    Vectorise les chunks médicaux avec BioBERT
    et les stocke dans ChromaDB.

    Pourquoi BioBERT ?
    BioBERT est un BERT pré-entraîné sur PubMed et
    PMC — il comprend le vocabulaire médical bien
    mieux qu'un modèle généraliste comme BERT classique.
    """

    def __init__(self):
        logger.info(f"Chargement du modèle : {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Modèle BioBERT chargé ✅")

        logger.info(f"Connexion ChromaDB : {CHROMA_HOST}:{CHROMA_PORT}")
        self.client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB connecté ✅")

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Crée ou récupère la collection ChromaDB.
        Une collection = une table dans une base vectorielle.
        """
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            count      = collection.count()
            logger.info(f"Collection '{COLLECTION_NAME}' existante — {count} vecteurs")
            return collection
        except Exception:
            collection = self.client.create_collection(
                name     = COLLECTION_NAME,
                metadata = {"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{COLLECTION_NAME}' créée ✅")
            return collection

    def load_chunks(self) -> list[dict]:
        """
        Charge le fichier de chunks le plus récent
        depuis data/processed/.
        """
        files = sorted(PROCESSED_DIR.glob("chunks_*.json"))
        if not files:
            raise FileNotFoundError("Aucun fichier chunks trouvé dans data/processed/")

        latest = files[-1]
        logger.info(f"Chargement : {latest}")

        with open(latest, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"{len(chunks)} chunks chargés")
        return chunks

    def embed_and_index(self, chunks: list[dict]) -> int:
        """
        Vectorise les chunks par batch et les indexe
        dans ChromaDB avec leurs métadonnées.

        Pourquoi par batch ?
        BioBERT est un modèle lourd — traiter chunk
        par chunk serait très lent. Par batch de 32,
        on parallélise le calcul sur le CPU/GPU.
        """
        total_indexed = 0

        # Découper en batches
        batches = [
            chunks[i:i + BATCH_SIZE]
            for i in range(0, len(chunks), BATCH_SIZE)
        ]

        logger.info(f"Indexation : {len(chunks)} chunks en {len(batches)} batches")

        for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding", unit="batch")):
            texts     = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            ids       = [
                f"{c['metadata']['source']}_{c['metadata'].get('report_id', c['metadata'].get('pmid', ''))}_{c['metadata']['chunk_index']}"
                for c in batch
            ]

            # Vectorisation avec BioBERT
            embeddings = self.model.encode(
                texts,
                batch_size      = BATCH_SIZE,
                show_progress_bar = False,
            ).tolist()

            # Nettoyage métadonnées
            # ChromaDB n'accepte que str, int, float, bool
            clean_metadatas = []
            for m in metadatas:
                clean_metadatas.append({
                    k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in m.items()
                })

            # Indexation dans ChromaDB
            try:
                self.collection.add(
                    embeddings = embeddings,
                    documents  = texts,
                    metadatas  = clean_metadatas,
                    ids        = ids,
                )
                total_indexed += len(batch)

            except Exception as e:
                logger.warning(f"  Batch {batch_idx} erreur : {e}")
                continue

        return total_indexed

    def verify_index(self) -> dict:
        """Vérifie que l'indexation s'est bien passée."""
        count = self.collection.count()
        logger.info(f"Vecteurs dans ChromaDB : {count}")

        # Test de recherche sémantique
        test_query  = "ibuprofen serious adverse reactions hospitalization"
        test_vector = self.model.encode(test_query).tolist()

        results = self.collection.query(
            query_embeddings = [test_vector],
            n_results        = 3,
        )

        logger.info("Test de recherche sémantique :")
        for i, doc in enumerate(results["documents"][0]):
            logger.info(f"  Résultat {i+1} : {doc[:120]}...")

        return {"total_vectors": count}


def run_embedding():
    """Point d'entrée principal."""
    logger.info("PharmaRAG — Embedding & Indexation démarrés")

    embedder = PharmaEmbedder()
    chunks   = embedder.load_chunks()
    total    = embedder.embed_and_index(chunks)

    logger.info(f"Indexation terminée — {total} chunks indexés")

    stats = embedder.verify_index()
    logger.info(f"Vérification : {stats['total_vectors']} vecteurs dans ChromaDB ✅")

    return stats


if __name__ == "__main__":
    run_embedding()