import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ecommerce_knowledge_base"

# 1. Initialize the same enterprise embedding model used during ingestion
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Connect to the existing Qdrant instance
client = QdrantClient(url=QDRANT_URL)

vector_store = QdrantVectorStore(
    client=client, collection_name=COLLECTION_NAME, embedding=embeddings_model
)


def query_knowledge_base(user_query: str, top_k: int = 1):
    print(f"\n[RETRIEVAL] Processing user query: '{user_query}'")

    # 3. Perform a similarity search with raw relevance scores
    # score_threshold filters out vectors that don't pass a basic geometric confidence bar
    results_with_scores = vector_store.similarity_search_with_score(
        query=user_query, k=top_k
    )

    print(
        f"[RETRIEVAL] Found {len(results_with_scores)} matches matching the criteria.\n"
    )

    retrieved_context = ""
    for doc, score in results_with_scores:
        print(f"--- MATCH (Cosine Score: {score:.4f}) ---")
        print(doc.page_content.strip())
        print(f"Metadata payload: {doc.metadata}\n")

        # Accumulate the string contents for prompt injection
        retrieved_context += f"{doc.page_content}\n\n"

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
