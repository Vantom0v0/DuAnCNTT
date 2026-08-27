from components.config import EMBEDDING_MODEL, INDEX_PATH
import faiss
import os

def embedder(target: list):
    embedding_model = EMBEDDING_MODEL
    print("Embedding model loaded.")

    embeddings = embedding_model.encode(
        target,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print("Embeddings successfully created.")

    return embeddings

def get_index(documents: list):
    """Load FAISS vector index or build it if it doesn't exist."""

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print(f"Retrieved vector index at {INDEX_PATH}")
    else:
        embeddings = embedder(documents)
        
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
