from components.llm import generate_answer
from components.retrieval import retrieve
from components.reranker import rerank
from components.retrieval_data import initiate
from components.config import TOP_N, TOP_K, LLM_MODEL, client

def naive_rag_model(query, groundtruth = "NA", evaluation_mode = False, enable_rerank = False):
    corpus_documents, _, index, _ = initiate()

    if enable_rerank is False:
        retrieved = retrieve(query, index, TOP_N, corpus_documents)
    else:
        retrieved = retrieve(query, index, TOP_K, corpus_documents)
        
    chosen_docs = retrieved
    print("RETRIEVED DOCUMENTS")

    if enable_rerank is True:
        reranked = rerank(query, retrieved)
        chosen_docs = reranked
        print("Finished reranking")

    if evaluation_mode is False:
        for i, doc in enumerate(chosen_docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Similarity: {doc['score']:.4f}")
            if enable_rerank is True:
                print(f"Rerank score: {doc['rerank_score']:.4f}")
            print(doc["text"][:500])

    instruction = """
                    Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                    Hãy trả lời câu hỏi của người dùng CHỈ dựa trên thông tin trong phần tài liệu được cung cấp.
                    Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ rằng không tìm thấy đủ thông tin thay vì tự bịa thêm kiến thức.
                    Không sử dụng kiến thức bên ngoài tài liệu.
                """

    answer = generate_answer(query, chosen_docs, LLM_MODEL, client, instruction)

    if evaluation_mode is False:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print(answer) 

    return {
        "model_name": "naive_rag",
        "question": query,
        "ground_truth": groundtruth,
        "answer": answer,
        "retrieved": chosen_docs,
        "llm_call": 1,
        "iteration": 1
    }
