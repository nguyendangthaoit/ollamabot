import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from read_pdf import extract_raw_text_from_pdf

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "company_policy"

client = QdrantClient(url=QDRANT_URL)

# 1. Update to the Modern Enterprise Flagship Embedding Model
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


# 2. Production Schema Verification Loop (Fixing Deprecation Warning)
# --- CHANGE THIS INSIDE YOUR SCRIPT ---
def init_vector_collection():
    print(f"[INGEST] Verifying collection existence for '{COLLECTION_NAME}'...")

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(
            f"[INGEST] Collection exists. Dropping old instances for fresh initialization..."
        )
        client.delete_collection(collection_name=COLLECTION_NAME)

    # UPDATE THE SIZE PARAMETER TO MATCH THE 3072 VECTOR OUTPUT
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,  # Exact dimension layout match for gemini-embedding-2-preview
            distance=Distance.COSINE,
        ),
    )
    print(f"[INGEST] Clean collection '{COLLECTION_NAME}' created successfully.")


# 3. Clean Text Splitting and Upserting
def chunk_and_embed_data():
    raw_document_data = extract_raw_text_from_pdf("company_policy.pdf")

    print("[INGEST] Splitting structural mock inventory streams...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=40, separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(raw_document_data)
    print(f"[INGEST] Document divided into {len(chunks)} contextual units.")

    print("[INGEST] Launching vector embeddings matrix computing...")
    metadatas = [
        {"source": "company_policy_pdf", "chunk_index": i} for i in range(len(chunks))
    ]

    from langchain_qdrant import QdrantVectorStore

    # Load data directly into local Docker space via LangChain adapter
    QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        metadatas=metadatas,
    )

    print(
        "[INGEST] All text nodes mapped and safely synchronized inside Qdrant storage!"
    )


if __name__ == "__main__":
    init_vector_collection()
    chunk_and_embed_data()
