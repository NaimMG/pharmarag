"""
PharmaRAG — Monitoring des requêtes
Logger toutes les questions + temps de réponse
+ scores de retrieval dans une base SQLite.

Permet d'analyser :
- Les questions les plus fréquentes
- Les temps de réponse moyens
- La qualité du retrieval dans le temps
- Les dérives de performance
"""

import sqlite3
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from loguru import logger

# ── Configuration ───────────────────────────────────────
DB_PATH = Path("data/monitoring.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Initialisation de la base ───────────────────────────
def init_db():
    """Crée les tables SQLite si elles n'existent pas."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                question        TEXT    NOT NULL,
                answer_length   INTEGER,
                response_time_s REAL,
                num_sources     INTEGER,
                top_source      TEXT,
                top_drug        TEXT,
                top_score       REAL,
                status          TEXT    DEFAULT 'success'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date            TEXT PRIMARY KEY,
                total_queries   INTEGER DEFAULT 0,
                avg_response_s  REAL,
                avg_sources     REAL,
                error_count     INTEGER DEFAULT 0
            )
        """)

        conn.commit()
    logger.info(f"Base monitoring initialisée : {DB_PATH}")


# ── Context manager pour mesurer le temps ──────────────
@contextmanager
def timer():
    """Context manager qui mesure le temps d'exécution."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


# ── Classe principale ───────────────────────────────────
class QueryMonitor:
    """
    Monitor des requêtes PharmaRAG.
    Enregistre chaque requête dans SQLite pour analyse.
    """

    def __init__(self):
        init_db()
        logger.info("QueryMonitor initialisé ✅")

    def log_query(
        self,
        question:       str,
        answer:         str,
        sources:        list[dict],
        response_time:  float,
        status:         str = "success",
    ):
        """
        Enregistre une requête dans la base SQLite.

        Args:
            question      : question posée
            answer        : réponse générée
            sources       : chunks récupérés avec scores
            response_time : temps de réponse en secondes
            status        : 'success' ou 'error'
        """
        timestamp    = datetime.now().isoformat()
        top_source   = sources[0]["source"]   if sources else ""
        top_drug     = sources[0]["drug"]     if sources else ""
        top_score    = sources[0].get("similarity", 0.0) if sources else 0.0

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO queries
                    (timestamp, question, answer_length, response_time_s,
                     num_sources, top_source, top_drug, top_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    question,
                    len(answer),
                    round(response_time, 3),
                    len(sources),
                    top_source,
                    top_drug,
                    top_score,
                    status,
                ))
                conn.commit()

            logger.debug(
                f"Query loggée : '{question[:40]}...' "
                f"| {response_time:.2f}s | {len(sources)} sources"
            )

        except Exception as e:
            logger.error(f"Erreur monitoring : {e}")

    def get_stats(self) -> dict:
        """
        Retourne les statistiques globales de monitoring.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Stats globales
                row = conn.execute("""
                    SELECT
                        COUNT(*)                    as total_queries,
                        AVG(response_time_s)        as avg_response_s,
                        MIN(response_time_s)        as min_response_s,
                        MAX(response_time_s)        as max_response_s,
                        AVG(num_sources)            as avg_sources,
                        SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors
                    FROM queries
                """).fetchone()

                # Top 5 médicaments les plus demandés
                top_drugs = conn.execute("""
                    SELECT top_drug, COUNT(*) as count
                    FROM queries
                    WHERE top_drug != ''
                    GROUP BY top_drug
                    ORDER BY count DESC
                    LIMIT 5
                """).fetchall()

                # Dernières requêtes
                recent = conn.execute("""
                    SELECT timestamp, question, response_time_s, status
                    FROM queries
                    ORDER BY id DESC
                    LIMIT 10
                """).fetchall()

            return {
                "total_queries":   row[0] or 0,
                "avg_response_s":  round(row[1] or 0, 2),
                "min_response_s":  round(row[2] or 0, 2),
                "max_response_s":  round(row[3] or 0, 2),
                "avg_sources":     round(row[4] or 0, 1),
                "error_count":     row[5] or 0,
                "top_drugs":       [{"drug": r[0], "count": r[1]} for r in top_drugs],
                "recent_queries":  [
                    {
                        "timestamp":      r[0],
                        "question":       r[1][:60],
                        "response_time":  r[2],
                        "status":         r[3],
                    }
                    for r in recent
                ],
            }

        except Exception as e:
            logger.error(f"Erreur get_stats : {e}")
            return {}

    def get_slow_queries(self, threshold_s: float = 30.0) -> list[dict]:
        """Retourne les requêtes dont le temps dépasse le seuil."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT timestamp, question, response_time_s
                    FROM queries
                    WHERE response_time_s > ?
                    ORDER BY response_time_s DESC
                    LIMIT 20
                """, (threshold_s,)).fetchall()

            return [
                {
                    "timestamp":     r[0],
                    "question":      r[1][:80],
                    "response_time": r[2],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Erreur get_slow_queries : {e}")
            return []


# ── Instance globale ────────────────────────────────────
monitor = QueryMonitor()