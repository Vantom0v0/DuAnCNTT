from sentence_transformers import SentenceTransformer
import faiss
import os

INDEX_PATH = "./vector_storage/medical.index"

def get_embedding_model(embedding_model_name: str):
    return SentenceTransformer(embedding_model_name)

def embedder(embedding_model_name: str, target: list):
    print("\nLoading embedding model...")
    embedding_model = get_embedding_model(embedding_model_name)
    print("Embedding model loaded.")

    print("\nCreating embeddings...")
    embeddings = embedding_model.encode(
        target,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print("Embeddings successfully created.")

    return embeddings

def get_index(embedding_model_name: str, documents: list):
    """Load FAISS vector index or build it if it doesn't exist."""

    if os.path.exists(INDEX_PATH):
        print(f"Retrieving vector index at {INDEX_PATH}")
        index = faiss.read_index("./vector_storage/medical.index")
    else:
        embeddings = embedder(embedding_model_name, documents)
        
        print("Start building FAISS vector index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(
            embeddings.astype("float32")
        )

        faiss.write_index(index, INDEX_PATH)

        print(f"\nFAISS index created at {INDEX_PATH}")
        print("Indexed vectors:", index.ntotal)

    return index
