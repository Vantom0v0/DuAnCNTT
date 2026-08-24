from components.config import RERANKER_MODEL_NAME, TOP_N
from sentence_transformers import CrossEncoder

def rerank(query, retrieved_documents):
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    print("Calculating rerank score")
    pairs = [ [query, document["text"]] for document in retrieved_documents]
    scores = reranker.predict(pairs)
    print("Finished calculation")

    print("Sorting reranked documents")
    reranked = []
    for document, score in zip(retrieved_documents, scores):
        reranked_document = document.copy()
        reranked_document["rerank_score"] = float(score)
        reranked.append(reranked_document)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:TOP_N]
