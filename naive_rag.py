from llm import generate_answer
from retrieval import retrieve
from reranker import rerank
from initial_retrieval import initiate
from config import EMBEDDING_MODEL_NAME, TOP_K, LLM_MODEL, client

def naive_rag_model(query):
    corpus_documents, _, index, _ = initiate(query)

    print("Retrieving documents")
    retrieved = retrieve(EMBEDDING_MODEL_NAME, query, index, TOP_K, corpus_documents)
    print("RETRIEVED DOCUMENTS")

    print("Reranking the documents")
    reranked = rerank(query, retrieved)
    print("Finished reranking")

    for i, doc in enumerate(reranked):
        print(f"\n--- Document {i+1} ---")
        print(f"Similarity: {doc['score']:.4f}")
        print(f"Rerank score: {doc['rerank_score']:.4f}")
        print(doc["text"][:500])

    instruction = """
                    Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                    Hãy trả lời câu hỏi của người dùng CHỈ dựa trên thông tin trong phần tài liệu được cung cấp.
                    Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ rằng không tìm thấy đủ thông tin thay vì tự bịa thêm kiến thức.
                    Không sử dụng kiến thức bên ngoài tài liệu.
                """

    answer = generate_answer(query, reranked, LLM_MODEL, client, instruction)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    return {
        "query": query,
        "retrieved": retrieved,
        "answer": answer
    }
