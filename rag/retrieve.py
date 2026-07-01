import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "company_policy"

# 1. Initialize the same enterprise embedding model used during ingestion
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Connect to the existing Qdrant instance
client = QdrantClient(url=QDRANT_URL)

vector_store = QdrantVectorStore(
    client=client, collection_name=COLLECTION_NAME, embedding=embeddings_model
)

flash_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")


def query_knowledge_base(user_query: str, top_k: int = 2):
    print(f"\n[RETRIEVAL] Processing user query: '{user_query}'")

    # Step 1: Cast a wider net from Qdrant to capture all potential data nodes
    raw_results = vector_store.similarity_search(query=user_query, k=10)

    if not raw_results:
        print("[RETRIEVAL] Zero initial matches found inside Qdrant cluster.")
        return ""

    # Step 2: Format LangChain objects into raw dict arrays required by FlashRank
    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(raw_results)
    ]

    # Step 3: Package execution request payload
    rerank_request = RerankRequest(query=user_query, passages=passages)

    print(
        f"[RERANKER] Processing {len(passages)} chunks through local Cross-Encoder matrix..."
    )
    reranked_results = flash_ranker.rerank(rerank_request)

    # Step 4: Extract and build the final filtered context string
    print(f"[RERANKER] Filtering out top {top_k} high-precision nodes.")

    retrieved_context = ""
    # Select only the top_k elements returned by the ranker execution
    for high_precision_node in reranked_results[:top_k]:
        score = high_precision_node["score"]
        text_content = high_precision_node["text"]

        print(f" -> Re-Ranked Match Confidence Score: {score:.4f}")
        retrieved_context += f"{text_content}\n\n"

    return retrieved_context


if __name__ == "__main__":
    # Test Scenario A: Semantic match (using synonyms like "dev" and "keyboards")
    context_a = query_knowledge_base(
        "Do you have any keyboards layout optimized for dev?", top_k=1
    )
    # Test Scenario B: Semantic match (asking about prolonged working hours/chairs)
    context_b = query_knowledge_base(
        "What items do you recommend for long sitting hours?", top_k=1
    )
