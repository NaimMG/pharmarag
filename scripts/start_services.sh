#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║   PharmaRAG — Démarrage rapide des services          ║
# ║   Usage : bash scripts/start_services.sh             ║
# ╚══════════════════════════════════════════════════════╝

echo "🚀 PharmaRAG — Démarrage des services"
echo "────────────────────────────────────"

# 1. ChromaDB
echo "1️⃣  Démarrage ChromaDB..."
docker start pharmarag-chromadb 2>/dev/null || \
docker run -d \
  --name pharmarag-chromadb \
  -p 8001:8000 \
  -v $(pwd)/data/embeddings:/chroma/chroma \
  chromadb/chroma:0.5.3
echo "✅ ChromaDB démarré sur port 8001"

# 2. Vérification Ollama
echo "2️⃣  Vérification Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama disponible sur port 11434"
else
    echo "⚠️  Ollama n'est pas lancé — démarre l'app Ollama manuellement"
fi

echo ""
echo "────────────────────────────────────"
echo "Services prêts ! Lance maintenant :"
echo ""
echo "  Terminal 1 (API) :"
echo "  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Terminal 2 (UI) :"
echo "  streamlit run src/ui/app.py"
echo "────────────────────────────────────"