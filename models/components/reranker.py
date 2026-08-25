from components.config import RERANK_MODEL, TOP_N

def rerank(query, retrieved_documents, return_amount = TOP_N, allow_sorting = True):
    reranker = RERANK_MODEL

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

    if allow_sorting is True:
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)


    return reranked[:return_amount]
