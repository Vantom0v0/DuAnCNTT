from data_construct import embedder

def retrieve(EMBEDDING_MODEL_NAME, query, vector_index, top_k, documents):
    query_embedding = embedder(EMBEDDING_MODEL_NAME, [query])

    scores, indices = vector_index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        
        results.append({
            "text": documents[idx]["text"],
            "score": float(score),
            "id": documents[idx]["id"]
        })

    return results