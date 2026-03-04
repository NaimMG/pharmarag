"""
PharmaRAG — Tests API FastAPI
Lance avec : pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ── Mock de PharmaRAGChain ──────────────────────────────
mock_rag_instance = MagicMock()
mock_rag_instance.collection.count.return_value = 1503
mock_rag_instance.query.return_value = {
    "question": "What are adverse reactions for ibuprofen?",
    "answer":   "Based on FDA FAERS data, ibuprofen is associated with gastrointestinal bleeding.",
    "sources": [
        {
            "source":    "faers",
            "drug":      "ibuprofen",
            "report_id": "12345",
            "pmid":      "",
            "text":      "Adverse Event Report — FDA FAERS. Adverse reactions: Gastrointestinal bleeding.",
        },
        {
            "source":    "pubmed",
            "drug":      "ibuprofen",
            "report_id": "",
            "pmid":      "23163543",
            "text":      "The relative safety of OTC ibuprofen has been supported by large-scale studies.",
        },
    ],
}


@pytest.fixture
def client():
    """
    Fixture pytest : crée un client de test FastAPI
    en mockant PharmaRAGChain au niveau du module.
    """
    with patch(
        "src.generation.rag_chain.PharmaRAGChain",
        return_value=mock_rag_instance
    ):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ── Tests /health ───────────────────────────────────────

def test_health_returns_200(client):
    """L'endpoint /health doit retourner 200."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_structure(client):
    """La réponse /health doit avoir les bons champs."""
    data = client.get("/health").json()
    assert "status"         in data
    assert "chroma_vectors" in data
    assert "llm_model"      in data
    assert "collection"     in data


def test_health_status_ok(client):
    """/health doit retourner status: ok."""
    assert client.get("/health").json()["status"] == "ok"


def test_health_chroma_vectors(client):
    """/health doit retourner le bon nombre de vecteurs."""
    assert client.get("/health").json()["chroma_vectors"] == 1503


# ── Tests /query ────────────────────────────────────────

def test_query_returns_200(client):
    """L'endpoint /query doit retourner 200."""
    resp = client.post(
        "/query",
        json={"question": "What are adverse reactions for ibuprofen?"}
    )
    assert resp.status_code == 200


def test_query_response_structure(client):
    """La réponse /query doit avoir les bons champs."""
    data = client.post(
        "/query",
        json={"question": "What are adverse reactions for ibuprofen?"}
    ).json()
    assert "question"      in data
    assert "answer"        in data
    assert "sources"       in data
    assert "total_sources" in data


def test_query_returns_sources(client):
    """/query doit retourner des sources."""
    data = client.post(
        "/query",
        json={"question": "What are adverse reactions for ibuprofen?"}
    ).json()
    assert data["total_sources"] > 0
    assert len(data["sources"])  > 0


def test_query_source_structure(client):
    """Chaque source doit avoir les bons champs."""
    source = client.post(
        "/query",
        json={"question": "What are adverse reactions for ibuprofen?"}
    ).json()["sources"][0]
    assert "source"    in source
    assert "drug"      in source
    assert "report_id" in source
    assert "pmid"      in source
    assert "text"      in source


def test_query_empty_question_returns_400(client):
    """Une question vide doit retourner 400."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 400


def test_query_too_long_returns_400(client):
    """Une question trop longue doit retourner 400."""
    resp = client.post("/query", json={"question": "a" * 501})
    assert resp.status_code == 400


# ── Tests /stats ────────────────────────────────────────

def test_stats_returns_200(client):
    """L'endpoint /stats doit retourner 200."""
    assert client.get("/stats").status_code == 200


def test_stats_response_structure(client):
    """La réponse /stats doit avoir les bons champs."""
    data = client.get("/stats").json()
    assert "total_vectors" in data
    assert "collection"    in data