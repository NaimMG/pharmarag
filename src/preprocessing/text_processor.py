"""
PharmaRAG — Preprocessing & Chunking
Charge les données brutes FAERS + PubMed,
nettoie le texte médical et découpe en chunks
prêts à être vectorisés.
"""

import json
import re
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Configuration ───────────────────────────────────────
RAW_FAERS_DIR   = Path("data/raw/faers")
RAW_PUBMED_DIR  = Path("data/raw/pubmed")
PROCESSED_DIR   = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50


class TextProcessor:
    """
    Nettoie et découpe les textes médicaux en chunks.

    Pourquoi RecursiveCharacterTextSplitter ?
    C'est le splitter recommandé par LangChain pour les textes
    généraux : il essaie de couper aux paragraphes d'abord,
    puis aux phrases, puis aux mots — pour garder le sens intact.
    """

    def __init__(
        self,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size        = chunk_size,
            chunk_overlap     = chunk_overlap,
            length_function   = len,
            separators        = ["\n\n", "\n", ". ", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """
        Nettoie un texte médical brut.
        Supprime les caractères parasites tout en
        conservant la ponctuation médicale utile.
        """
        if not text or not isinstance(text, str):
            return ""

        # Supprimer caractères non imprimables
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)

        # Normaliser les espaces multiples
        text = re.sub(r" {2,}", " ", text)

        # Normaliser les sauts de ligne multiples
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Supprimer les lignes vides en début/fin
        text = text.strip()

        return text

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """
        Découpe un texte en chunks avec ses métadonnées.

        Les métadonnées sont cruciales : elles permettront
        de filtrer les résultats dans ChromaDB (ex: filtrer
        par médicament, par source, par date).
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        chunks = self.splitter.split_text(cleaned)

        return [
            {
                "text":       chunk,
                "metadata":   {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip()  # Ignorer les chunks vides
        ]

    # ── Loaders ─────────────────────────────────────────

    def load_faers_file(self, filepath: Path) -> list[dict]:
        """Charge et chunke un fichier JSON FAERS."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        drug    = data.get("drug", "unknown")
        reports = data.get("reports", [])
        chunks  = []

        for report in reports:
            full_text = report.get("full_text", "")
            if not full_text:
                continue

            metadata = {
                "source":      "faers",
                "drug":        drug,
                "report_id":   report.get("report_id", ""),
                "report_date": report.get("report_date", ""),
                "country":     report.get("country", ""),
                "patient_age": report.get("patient_age", ""),
                "patient_sex": report.get("patient_sex", ""),
                "serious":     ", ".join(report.get("serious_criteria", [])),
            }

            chunks.extend(self.chunk_text(full_text, metadata))

        return chunks

    def load_pubmed_file(self, filepath: Path) -> list[dict]:
        """Charge et chunke un fichier JSON PubMed."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        chunks   = []

        for article in articles:
            full_text = article.get("full_text", "")
            if not full_text:
                continue

            metadata = {
                "source":  "pubmed",
                "pmid":    article.get("pmid", ""),
                "title":   article.get("title", "")[:100],
                "journal": article.get("journal", ""),
                "year":    article.get("year", ""),
                "drug":    data.get("query", "").split()[0],
            }

            chunks.extend(self.chunk_text(full_text, metadata))

        return chunks

    # ── Pipeline principal ───────────────────────────────

    def process_all(self) -> dict:
        """
        Charge tous les fichiers bruts, les chunke
        et sauvegarde le résultat dans data/processed/.
        """
        all_chunks = []

        # ── FAERS ──────────────────────────────────────
        faers_files = list(RAW_FAERS_DIR.glob("*.json"))
        logger.info(f"Fichiers FAERS trouvés : {len(faers_files)}")

        for filepath in tqdm(faers_files, desc="FAERS", unit="file"):
            chunks = self.load_faers_file(filepath)
            all_chunks.extend(chunks)
            logger.info(f"  {filepath.name} → {len(chunks)} chunks")

        # ── PubMed ─────────────────────────────────────
        pubmed_files = list(RAW_PUBMED_DIR.glob("*.json"))
        logger.info(f"Fichiers PubMed trouvés : {len(pubmed_files)}")

        for filepath in tqdm(pubmed_files, desc="PubMed", unit="file"):
            chunks = self.load_pubmed_file(filepath)
            all_chunks.extend(chunks)
            logger.info(f"  {filepath.name} → {len(chunks)} chunks")

        # ── Sauvegarde ─────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output    = PROCESSED_DIR / f"chunks_{timestamp}.json"

        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        stats = {
            "total_chunks":  len(all_chunks),
            "faers_files":   len(faers_files),
            "pubmed_files":  len(pubmed_files),
            "output_file":   str(output),
        }

        logger.info("─" * 45)
        logger.info(f"✅ Preprocessing terminé")
        logger.info(f"   Total chunks  : {stats['total_chunks']}")
        logger.info(f"   Fichier sortie : {output}")

        return stats


def run_preprocessing():
    """Point d'entrée principal."""
    logger.info("PharmaRAG — Preprocessing démarré")
    processor = TextProcessor()
    stats     = processor.process_all()
    return stats


if __name__ == "__main__":
    run_preprocessing()