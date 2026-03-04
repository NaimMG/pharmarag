"""
PharmaRAG — FastAPI Backend
Expose la RAG Chain via une API REST.

Endpoints :
  GET  /health  → statut de l'API
  POST /query   → question → réponse + sources
  GET  /stats   → statistiques ChromaDB
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.generation.rag_chain import PharmaRAGChain

from src.api.monitoring import monitor, timer
import time
# ── Modèles Pydantic ────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k:    int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the serious adverse reactions for ibuprofen?",
                "top_k":    5,
            }
        }


class SourceModel(BaseModel):
    source:    str
    drug:      str
    report_id: str
    pmid:      str
    text:      str


class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[SourceModel]
    total_sources: int


class HealthResponse(BaseModel):
    status:     str
    chroma_vectors: int
    llm_model:  str
    collection: str


class StatsResponse(BaseModel):
    total_vectors: int
    collection:    str


# ── Lifespan (startup/shutdown) ─────────────────────────

rag_chain: PharmaRAGChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise la RAG Chain au démarrage de l'API.
    Le lifespan remplace l'ancien @app.on_event("startup")
    dans les versions récentes de FastAPI.
    """
    global rag_chain
    logger.info("Démarrage API — Initialisation RAG Chain...")
    try:
        rag_chain = PharmaRAGChain()
        logger.info("RAG Chain initialisée ✅")
    except Exception as e:
        logger.error(f"Erreur initialisation RAG Chain : {e}")
        raise

    yield  # L'API tourne ici

    logger.info("Arrêt API")


# ── Application FastAPI ─────────────────────────────────

app = FastAPI(
    title       = "PharmaRAG API",
    description = "RAG system for pharmacovigilance — FDA FAERS + PubMed",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Endpoints ───────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie que l'API et tous ses composants sont opérationnels."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG Chain non initialisée")

    try:
        count = rag_chain.collection.count()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB inaccessible : {e}")

    return HealthResponse(
        status          = "ok",
        chroma_vectors  = count,
        llm_model       = "llama3.2",
        collection      = "pharmavigilance",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG Chain non initialisée")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Question trop longue (max 500 caractères)")

    try:
        logger.info(f"Query reçue : {request.question}")

        start_time = time.time()
        result     = rag_chain.query(request.question)
        elapsed    = time.time() - start_time

        sources = [
            SourceModel(
                source    = s.get("source", ""),
                drug      = s.get("drug", ""),
                report_id = s.get("report_id", ""),
                pmid      = s.get("pmid", ""),
                text      = s.get("text", "")[:300],
            )
            for s in result["sources"]
        ]

        # Log dans SQLite
        monitor.log_query(
            question      = request.question,
            answer        = result["answer"],
            sources       = result["sources"],
            response_time = elapsed,
            status        = "success",
        )

        return QueryResponse(
            question      = result["question"],
            answer        = result["answer"],
            sources       = sources,
            total_sources = len(sources),
        )

    except Exception as e:
        monitor.log_query(
            question      = request.question,
            answer        = "",
            sources       = [],
            response_time = 0.0,
            status        = "error",
        )
        logger.error(f"Erreur query : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Retourne les statistiques de la base vectorielle ChromaDB."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG Chain non initialisée")

    try:
        count = rag_chain.collection.count()
        return StatsResponse(
            total_vectors = count,
            collection    = "pharmavigilance",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/monitoring")
async def monitoring_stats():
    """Retourne les statistiques de monitoring des requêtes."""
    return monitor.get_stats()


# ── Point d'entrée direct ───────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,
        log_level = "info",
    )