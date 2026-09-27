"""
Test script for live ChromaDB retrieval and end-to-end RAG pipeline.
"""

import sys
from pathlib import Path
import time

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.services.retriever import get_retriever
from app.services.rag_service import RAGService

retriever = get_retriever()

print("=================================================================")
print("TEST 1: NORMAL QUERY (Technology)")
print("=================================================================")
q1 = "Apple unveils new MacBook Pro with M3 Max chips, high performance GPU, and neural engine architecture."
t0 = time.time()
hits1 = retriever.retrieve(collection="rad_llm_n24_news", query_text=q1, k=3, where={"is_anomaly": 0})
t1 = time.time()
print(f"Retrieval completed in {(t1 - t0)*1000:.2f} ms")
for i, h in enumerate(hits1):
    cat = h["metadata"].get("category")
    doc_id = h["metadata"].get("doc_id")
    dist = h["distance"]
    print(f"Hit {i+1}: ID={doc_id}, Category={cat}, Distance={dist:.4f}")
    print(f"  Snippet: {h['text'][:160]}...\n")

print("=================================================================")
print("TEST 2: ANOMALY QUERY (Food / Recipe)")
print("=================================================================")
q2 = "Mix active sourdough starter with unbleached flour, water, and sea salt. Let ferment overnight in a Dutch oven for crispy crust."
t0 = time.time()
hits2 = retriever.retrieve(collection="rad_llm_n24_news", query_text=q2, k=3, where={"is_anomaly": 0})
t1 = time.time()
print(f"Retrieval completed in {(t1 - t0)*1000:.2f} ms")
for i, h in enumerate(hits2):
    cat = h["metadata"].get("category")
    doc_id = h["metadata"].get("doc_id")
    dist = h["distance"]
    print(f"Hit {i+1}: ID={doc_id}, Category={cat}, Distance={dist:.4f}")
    print(f"  Snippet: {h['text'][:160]}...\n")

print("=================================================================")
print("TEST 3: FULL RAG ORCHESTRATION PIPELINE")
print("=================================================================")
service = RAGService(retriever=retriever)
result = service.run(query=q1, collection="rad_llm_n24_news", k=2, where={"is_anomaly": 0})
print("Generated RAG Prompt Preview:\n")
print(result.prompt)
print("-----------------------------------------------------------------")
print("LLM Client Response:\n")
print(result.response.text)
