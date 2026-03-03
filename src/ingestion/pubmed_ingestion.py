"""
PharmaRAG — Ingestion PubMed / NCBI
Source : Entrez API (NCBI) — gratuite, sans clé requise
Doc    : https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import json
import time
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm
from loguru import logger

# ── Configuration ───────────────────────────────────────
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RAW_DATA_DIR      = Path("data/raw/pubmed")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Requêtes ciblées pharmacovigilance
PUBMED_QUERIES = [
    "ibuprofen adverse effects",
    "metformin side effects",
    "atorvastatin adverse reactions",
    "amoxicillin adverse effects",
    "aspirin adverse reactions",
]


class PubMedIngestion:
    """
    Collecte d'abstracts scientifiques depuis PubMed NCBI.
    Utilise l'API Entrez — gratuite, max 3 requêtes/seconde.
    """

    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self.session     = requests.Session()
        self.session.headers.update({
            "User-Agent": "PharmaRAG/1.0 (academic research)"
        })

    def search_pmids(self, query: str) -> list[str]:
        """
        Étape 1 : cherche les PMIDs correspondant à une requête.
        PMID = identifiant unique d'un article PubMed.
        """
        params = {
            "db":      "pubmed",
            "term":    query,
            "retmax":  self.max_results,
            "retmode": "json",
            "sort":    "relevance",
        }

        try:
            resp = self.session.get(PUBMED_SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data  = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"  '{query}' → {len(pmids)} articles trouvés")
            return pmids

        except requests.exceptions.RequestException as e:
            logger.error(f"  Erreur recherche PubMed : {e}")
            return []

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """
        Étape 2 : récupère les abstracts en batch
        pour une liste de PMIDs.
        """
        if not pmids:
            return []

        params = {
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        try:
            resp = self.session.get(PUBMED_FETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            return self._parse_xml(resp.text)

        except requests.exceptions.RequestException as e:
            logger.error(f"  Erreur fetch abstracts : {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[dict]:
        """
        Parse le XML retourné par PubMed.
        Extrait : titre, abstract, journal, année, auteurs.
        """
        articles = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"  Erreur parsing XML : {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                # PMID
                pmid_el = article.find(".//PMID")
                pmid    = pmid_el.text if pmid_el is not None else ""

                # Titre
                title_el = article.find(".//ArticleTitle")
                title    = title_el.text or "" if title_el is not None else ""

                # Abstract (peut avoir plusieurs sections)
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    (
                        (el.get("Label", "") + ": " if el.get("Label") else "")
                        + (el.text or "")
                    ).strip()
                    for el in abstract_parts
                    if el.text
                )

                # Ignorer les articles sans abstract
                if not title or not abstract:
                    continue

                # Journal
                journal_el = article.find(".//Journal/Title")
                journal    = journal_el.text if journal_el is not None else ""

                # Année
                year_el = article.find(".//PubDate/Year")
                year    = year_el.text if year_el is not None else ""

                # Auteurs (max 3)
                authors = [
                    f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip()
                    for a in article.findall(".//Author")[:3]
                ]

                full_text = f"""PubMed Article — PMID: {pmid}
Title: {title}
Authors: {', '.join(authors)}
Journal: {journal} ({year})

Abstract:
{abstract}""".strip()

                articles.append({
                    "pmid":      pmid,
                    "title":     title,
                    "abstract":  abstract,
                    "journal":   journal,
                    "year":      year,
                    "authors":   authors,
                    "source":    "pubmed",
                    "full_text": full_text,
                })

            except Exception as e:
                logger.warning(f"  Erreur parsing article : {e}")
                continue

        return articles

    def save_articles(self, query: str, articles: list[dict]) -> Path:
        """Sauvegarde les articles en JSON dans data/raw/pubmed/."""
        safe_query = query[:40].replace(" ", "_")
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = RAW_DATA_DIR / f"pubmed_{safe_query}_{timestamp}.json"

        payload = {
            "query":        query,
            "collected_at": timestamp,
            "total":        len(articles),
            "articles":     articles,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"  Sauvegardé : {filename}")
        return filename


def run_pubmed_ingestion(max_results: int = 50):
    """Point d'entrée principal."""
    logger.info("PharmaRAG — Ingestion PubMed démarrée")
    logger.info(f"Requêtes : {PUBMED_QUERIES}")

    client = PubMedIngestion(max_results=max_results)
    total  = 0

    for query in tqdm(PUBMED_QUERIES, desc="PubMed", unit="query"):
        pmids    = client.search_pmids(query)
        articles = client.fetch_abstracts(pmids)

        if articles:
            client.save_articles(query, articles)
            total += len(articles)

        # Respecter limite NCBI : max 3 requêtes/seconde
        time.sleep(0.4)

    logger.info(f"Ingestion PubMed terminée — {total} articles collectés")
    return total


if __name__ == "__main__":
    run_pubmed_ingestion()