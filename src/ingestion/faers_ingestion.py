"""
PharmaRAG — Ingestion FDA FAERS
Source  : https://api.fda.gov/drug/event.json
Doc API : https://open.fda.gov/apis/drug/event/
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

import requests
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────
FAERS_BASE_URL = os.getenv("FAERS_BASE_URL", "https://api.fda.gov/drug/event.json")
RAW_DATA_DIR   = Path("data/raw/faers")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Médicaments ciblés
TARGET_DRUGS = [
    "ibuprofen",
    "metformin",
    "atorvastatin",
    "amoxicillin",
    "aspirin",
]


class FAERSIngestion:
    """
    Collecte les rapports d'effets indésirables
    depuis l'API publique FDA openFDA (gratuite).
    """

    def __init__(self, limit_per_drug: int = 100):
        self.limit_per_drug = limit_per_drug
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PharmaRAG/1.0 (academic research)"
        })

    def build_url(self, drug: str, skip: int = 0) -> str:
        """Construit l'URL de requête FDA FAERS."""
        return (
            f"{FAERS_BASE_URL}"
            f"?search=patient.drug.medicinalproduct:\"{drug}\""
            f"+AND+serious:1"
            f"&limit=100"
            f"&skip={skip}"
        )

    def fetch_reports(self, drug: str) -> list[dict]:
        """
        Récupère tous les rapports pour un médicament.
        Gère la pagination automatiquement.
        """
        all_reports = []
        skip        = 0
        total       = None

        logger.info(f"Collecte rapports pour : {drug.upper()}")

        while True:
            url = self.build_url(drug, skip=skip)

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    logger.warning(f"Aucun résultat pour {drug}")
                    break
                logger.error(f"HTTP {resp.status_code} pour {drug}: {e}")
                break

            except requests.exceptions.RequestException as e:
                logger.error(f"Requête échouée pour {drug}: {e}")
                break

            if total is None:
                total = data.get("meta", {}).get("results", {}).get("total", 0)
                logger.info(f"  Total disponible : {total:,} rapports")

            results = data.get("results", [])
            if not results:
                break

            all_reports.extend(results)
            skip += len(results)

            if skip >= self.limit_per_drug or skip >= total:
                break

            time.sleep(0.3)

        logger.info(f"  {len(all_reports)} rapports collectés pour {drug}")
        return all_reports

    def extract_text_fields(self, report: dict) -> dict:
        """
        Extrait et structure les champs clés d'un rapport FAERS.
        C'est ce texte qui sera indexé dans ChromaDB.
        """
        patient   = report.get("patient", {})
        drugs     = patient.get("drug", [])
        reactions = patient.get("reaction", [])

        drug_names = [
            d.get("medicinalproduct", "").strip()
            for d in drugs
            if d.get("medicinalproduct")
        ]

        reaction_terms = [
            r.get("reactionmeddrapt", "").strip()
            for r in reactions
            if r.get("reactionmeddrapt")
        ]

        outcomes_map = {
            "1": "Recovered",
            "2": "Recovering",
            "3": "Not Recovered",
            "4": "Recovered with Sequelae",
            "5": "Fatal",
            "6": "Unknown",
        }
        reaction_outcomes = [
            outcomes_map.get(r.get("reactionoutcome", ""), "Unknown")
            for r in reactions
        ]

        age     = patient.get("patientonsetage", "Unknown")
        sex_map = {"1": "Male", "2": "Female", "0": "Unknown"}
        sex     = sex_map.get(str(patient.get("patientsex", "0")), "Unknown")

        serious_criteria = []
        if report.get("seriousnessdeath")           == "1": serious_criteria.append("Death")
        if report.get("seriousnesslifethreatening") == "1": serious_criteria.append("Life-threatening")
        if report.get("seriousnesshospitalization") == "1": serious_criteria.append("Hospitalization")
        if report.get("seriousnessdisabling")       == "1": serious_criteria.append("Disabling")

        full_text = self._build_full_text(
            drug_names, reaction_terms, reaction_outcomes,
            serious_criteria, str(age), sex,
            report.get("receiptdate", ""),
            report.get("primarysourcecountry", ""),
        )

        return {
            "report_id":        report.get("safetyreportid", ""),
            "report_date":      report.get("receiptdate", ""),
            "country":          report.get("primarysourcecountry", ""),
            "serious_criteria": serious_criteria,
            "patient_age":      str(age),
            "patient_sex":      sex,
            "drugs":            drug_names,
            "reactions":        reaction_terms,
            "outcomes":         reaction_outcomes,
            "full_text":        full_text,
        }

    def _build_full_text(
        self,
        drugs: list[str],
        reactions: list[str],
        outcomes: list[str],
        serious: list[str],
        age: str,
        sex: str,
        date: str,
        country: str,
    ) -> str:
        """
        Construit le texte narratif médical structuré
        qui sera découpé en chunks puis vectorisé.
        """
        reactions_str = "\n".join(
            f"- {r} (outcome: {o})"
            for r, o in zip(reactions, outcomes)
        ) if reactions else "- Not specified"

        return f"""Adverse Event Report — FDA FAERS
Date: {date} | Country: {country}
Patient: {sex}, age {age}
Suspected drugs: {', '.join(drugs) if drugs else 'Not specified'}
Adverse reactions:
{reactions_str}
Seriousness: {', '.join(serious) if serious else 'Serious (unspecified)'}""".strip()

    def save_reports(self, drug: str, reports: list[dict]) -> Path:
        """Sauvegarde les rapports traités en JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = RAW_DATA_DIR / f"faers_{drug}_{timestamp}.json"
        processed = [self.extract_text_fields(r) for r in reports]

        payload = {
            "drug":         drug,
            "collected_at": timestamp,
            "total":        len(processed),
            "reports":      processed,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"  Sauvegardé : {filename}")
        return filename


def run_ingestion(limit_per_drug: int = 100):
    """Point d'entrée principal."""
    logger.info("PharmaRAG — Ingestion FDA FAERS démarrée")
    logger.info(f"Médicaments : {TARGET_DRUGS}")

    client  = FAERSIngestion(limit_per_drug=limit_per_drug)
    summary = []

    for drug in tqdm(TARGET_DRUGS, desc="FAERS", unit="drug"):
        reports = client.fetch_reports(drug)
        if reports:
            path = client.save_reports(drug, reports)
            summary.append({"drug": drug, "count": len(reports)})
        time.sleep(1)

    total = sum(s["count"] for s in summary)
    logger.info("Ingestion terminée")
    logger.info(f"TOTAL : {total} rapports collectés dans data/raw/faers/")
    return summary


if __name__ == "__main__":
    run_ingestion(limit_per_drug=100)