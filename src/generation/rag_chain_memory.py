"""
PharmaRAG — RAG Chain avec Mémoire Conversationnelle
Étend la RAG Chain de base avec ConversationBufferMemory
pour permettre des questions de suivi contextuelles.

Exemple :
  Q1: "What are the side effects of ibuprofen?"
  Q2: "Are any of them life-threatening?"  ← comprend "them" = side effects of ibuprofen
  Q3: "What about in elderly patients?"    ← comprend le contexte complet
"""

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from loguru import logger

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ───────────────────────────────────────
CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8001
COLLECTION_NAME = "pharmavigilance"
EMBEDDING_MODEL = "dmis-lab/biobert-base-cased-v1.2"
LLM_MODEL       = "llama3.2"
TOP_K           = 5
MEMORY_WINDOW   = 3  # Nombre de tours de conversation mémorisés


# ── Prompt avec historique ───────────────────────────────
MEMORY_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""You are a pharmacovigilance expert assistant with memory of our conversation.
Your role is to analyze adverse drug event reports from FDA FAERS and PubMed.

CONVERSATION HISTORY:
{history}

CONTEXT FROM FDA FAERS AND PUBMED:
{context}

IMPORTANT RULES:
- Use the conversation history to understand follow-up questions
- Answer ONLY based on the provided context and conversation history
- Always cite your sources (FDA FAERS report ID or PubMed PMID)
- Never provide medical advice or diagnoses
- Be precise and factual

CURRENT QUESTION: {question}

ANSWER:"""
)


class PharmaRAGChainWithMemory:
    """
    RAG Chain avec mémoire conversationnelle.

    ConversationBufferWindowMemory garde les N derniers
    tours de conversation (paramètre MEMORY_WINDOW=3).
    Cela évite que le contexte devienne trop long tout en
    gardant les informations récentes pertinentes.
    """

    def __init__(self):
        # BioBERT
        logger.info("Chargement BioBERT...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB
        logger.info("Connexion ChromaDB...")
        self.chroma_client = chromadb.HttpClient(
            host     = CHROMA_HOST,
            port     = CHROMA_PORT,
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' — {self.collection.count()} vecteurs ✅")

        # Llama3.2
        logger.info("Connexion Ollama...")
        self.llm = Ollama(model=LLM_MODEL, temperature=0.1)

        # Mémoire conversationnelle
        self.memory = ConversationBufferWindowMemory(
            k                  = MEMORY_WINDOW,
            return_messages    = False,
            human_prefix       = "User",
            ai_prefix          = "Assistant",
        )

        # Chaîne LangChain
        self.chain = MEMORY_PROMPT | self.llm | StrOutputParser()

        logger.info(f"RAG Chain avec mémoire prête ✅ (fenêtre: {MEMORY_WINDOW} tours)")

    def retrieve(self, question: str) -> list[dict]:
        """Recherche sémantique dans ChromaDB."""
        query_vector = self.embedder.encode(question).tolist()
        results      = self.collection.query(
            query_embeddings = [query_vector],
            n_results        = TOP_K,
        )
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            chunks.append({
                "text":      doc,
                "source":    meta.get("source", ""),
                "drug":      meta.get("drug", ""),
                "report_id": meta.get("report_id", ""),
                "pmid":      meta.get("pmid", ""),
            })
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """Formate les chunks en contexte lisible."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            if chunk["source"] == "faers":
                ref = f"[FDA FAERS — Drug: {chunk['drug']} | Report: {chunk['report_id']}]"
            else:
                ref = f"[PubMed — PMID: {chunk['pmid']} | Drug: {chunk['drug']}]"
            parts.append(f"Source {i} {ref}:\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str) -> dict:
        """
        Pose une question en tenant compte de l'historique.
        La mémoire est automatiquement mise à jour après chaque réponse.
        """
        logger.info(f"Question : {question}")

        # Récupérer l'historique
        history = self.memory.load_memory_variables({}).get("history", "")

        # Retrieval
        chunks  = self.retrieve(question)
        context = self.format_context(chunks)
        logger.info(f"  {len(chunks)} chunks récupérés")

        # Génération avec historique
        logger.info("  Génération en cours...")
        answer = self.chain.invoke({
            "history":  history,
            "context":  context,
            "question": question,
        })

        # Sauvegarder dans la mémoire
        self.memory.save_context(
            {"input":  question},
            {"output": answer},
        )

        logger.info("  Réponse générée ✅")
        return {
            "question": question,
            "answer":   answer,
            "sources":  chunks,
            "history_length": len(self.memory.chat_memory.messages),
        }

    def clear_memory(self):
        """Efface l'historique de conversation."""
        self.memory.clear()
        logger.info("Mémoire effacée ✅")

    def get_history(self) -> str:
        """Retourne l'historique formaté."""
        return self.memory.load_memory_variables({}).get("history", "")


def run_conversation_test():
    """
    Test de conversation multi-tours pour valider la mémoire.
    """
    logger.info("PharmaRAG — Test conversation avec mémoire")
    logger.info("=" * 55)

    chain = PharmaRAGChainWithMemory()

    # Simulation d'une conversation réelle
    conversation = [
        "What are the adverse reactions reported for ibuprofen?",
        "Are any of them life-threatening?",
        "What about aspirin, does it have similar reactions?",
    ]

    for question in conversation:
        print(f"\n{'='*55}")
        print(f"❓ {question}")
        print(f"📝 Historique : {chain.get_history()[:100]}..." if chain.get_history() else "📝 Historique : (vide)")

        result = chain.query(question)

        print(f"\n💬 RÉPONSE:\n{result['answer']}")
        print(f"\n📚 Sources : {result['total_sources'] if 'total_sources' in result else len(result['sources'])}")
        print(f"🧠 Messages en mémoire : {result['history_length']}")


if __name__ == "__main__":
    run_conversation_test()