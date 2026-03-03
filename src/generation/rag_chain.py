"""
PharmaRAG — RAG Chain
Connecte ChromaDB + BioBERT + Llama3.2 via LangChain
pour répondre aux questions de pharmacovigilance.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from loguru import logger

# ── Configuration ───────────────────────────────────────
CHROMA_HOST      = "localhost"
CHROMA_PORT      = 8001
COLLECTION_NAME  = "pharmavigilance"
EMBEDDING_MODEL  = "dmis-lab/biobert-base-cased-v1.2"
LLM_MODEL        = "llama3.2"
TOP_K            = 5        # Nombre de chunks récupérés
TEMPERATURE      = 0.1      # Faible = réponses factuelles


# ── Prompt médical ──────────────────────────────────────
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a pharmacovigilance expert assistant.
Your role is to analyze adverse drug event reports from FDA FAERS
and scientific literature from PubMed.

IMPORTANT RULES:
- Answer ONLY based on the provided context
- Always cite your sources (FDA FAERS report ID or PubMed PMID)
- If the context does not contain enough information, say so clearly
- Never provide medical advice or diagnoses
- Be precise and factual

CONTEXT FROM FDA FAERS AND PUBMED:
{context}

QUESTION: {question}

ANSWER (based strictly on the context above):"""
)


class PharmaRAGChain:
    """
    Chaîne RAG complète pour la pharmacovigilance.

    Fonctionnement :
    1. La question est vectorisée avec BioBERT
    2. Les chunks les plus proches sont récupérés depuis ChromaDB
    3. La question + les chunks sont envoyés à Llama3.2
    4. Llama3.2 génère une réponse basée sur les sources
    """

    def __init__(self):
        # Chargement BioBERT
        logger.info(f"Chargement BioBERT : {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("BioBERT chargé ✅")

        # Connexion ChromaDB
        logger.info(f"Connexion ChromaDB : {CHROMA_HOST}:{CHROMA_PORT}")
        self.chroma_client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        count = self.collection.count()
        logger.info(f"Collection '{COLLECTION_NAME}' — {count} vecteurs ✅")

        # Connexion Ollama / Llama3.2
        logger.info(f"Connexion Ollama : {LLM_MODEL}")
        self.llm = Ollama(
            model       = LLM_MODEL,
            temperature = TEMPERATURE,
        )
        logger.info("Llama3.2 connecté ✅")

        # Construction de la chaîne LangChain
        self.chain = self._build_chain()
        logger.info("RAG Chain prête ✅")

    def retrieve(self, question: str) -> list[dict]:
        """
        Étape 1 — Retrieval :
        Vectorise la question et récupère les TOP_K
        chunks les plus sémantiquement proches.
        """
        query_vector = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings = [query_vector],
            n_results        = TOP_K,
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            chunks.append({
                "text":     doc,
                "source":   metadata.get("source", ""),
                "drug":     metadata.get("drug", ""),
                "report_id": metadata.get("report_id", ""),
                "pmid":     metadata.get("pmid", ""),
            })

        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """
        Formate les chunks récupérés en un contexte
        lisible pour le LLM avec les sources citées.
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk["source"].upper()
            if source == "FAERS":
                ref = f"[FDA FAERS — Drug: {chunk['drug']} | Report: {chunk['report_id']}]"
            else:
                ref = f"[PubMed — PMID: {chunk['pmid']} | Drug: {chunk['drug']}]"

            context_parts.append(f"Source {i} {ref}:\n{chunk['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _build_chain(self):
        """
        Construit la chaîne LangChain :
        prompt | llm | output_parser
        """
        return MEDICAL_PROMPT | self.llm | StrOutputParser()

    def query(self, question: str) -> dict:
        """
        Point d'entrée principal — pose une question
        et retourne la réponse avec les sources.
        """
        logger.info(f"Question : {question}")

        # Retrieval
        chunks  = self.retrieve(question)
        context = self.format_context(chunks)
        logger.info(f"  {len(chunks)} chunks récupérés depuis ChromaDB")

        # Generation
        logger.info("  Génération en cours (Llama3.2)...")
        answer = self.chain.invoke({
            "context":  context,
            "question": question,
        })

        logger.info("  Réponse générée ✅")

        return {
            "question": question,
            "answer":   answer,
            "sources":  chunks,
        }


def run_test():
    """
    Test rapide de la chaîne RAG complète
    avec 3 questions de pharmacovigilance.
    """
    logger.info("PharmaRAG — Test de la RAG Chain")
    logger.info("=" * 50)

    chain = PharmaRAGChain()

    test_questions = [
        "What are the most serious adverse reactions reported for ibuprofen?",
        "Are there any fatal outcomes associated with aspirin in the FDA reports?",
        "What does the scientific literature say about metformin side effects?",
    ]

    for question in test_questions:
        logger.info(f"\n{'=' * 50}")
        result = chain.query(question)

        print(f"\n❓ QUESTION:\n{result['question']}")
        print(f"\n💬 ANSWER:\n{result['answer']}")
        print(f"\n📚 SOURCES ({len(result['sources'])}) :")
        for s in result["sources"]:
            print(f"   - [{s['source'].upper()}] {s['drug']} | "
                  f"{s.get('report_id') or s.get('pmid', '')}")
        print("=" * 50)


if __name__ == "__main__":
    run_test()