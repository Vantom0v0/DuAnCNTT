from components.config import EMBEDDING_MODEL_NAME, TOP_N, LLM_MODEL, client
from components.retrieval_data import initiate
from components.retrieval import retrieve
from components.llm import generate_answer

def iter_retgen_model(query, iteration, evaluation_mode = False):
    corpus_documents, _, vector_index, _ = initiate(query)

    retrieval_info = query
    output_answer = ""
    llm_calls = 0
    num_iteration = 0
    for iteration_idx in range(iteration):
        print(f"Iteration {iteration_idx+1}:")
        retrieved_documents = retrieve(EMBEDDING_MODEL_NAME, retrieval_info, vector_index, TOP_N, corpus_documents)

        if evaluation_mode is False:
            for doc_idx, doc in enumerate(retrieved_documents):
                print(f"\n--- Document {doc_idx+1} ---")
                print(f"Similarity: {doc['score']:.4f}")
                print(doc["text"][:500])

        instruction = f"""
                        Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                        Hãy trả lời câu hỏi của người dùng dựa trên thông tin trong phần tài liệu được cung cấp.
                        Bạn phải suy nghĩ từng bước một trước khi đưa ra câu trả lời.
                    """

        answer = generate_answer(query, retrieved_documents, LLM_MODEL, client, instruction)
        if evaluation_mode is False:
            print("\n" + "=" * 70)
            print(f"Iteration {iteration_idx+1} answer:")
            print(answer)

        retrieval_info = query + "\n" + answer
        output_answer = answer

        llm_calls += 1
        num_iteration += 1

    return {
        "model_name": "iter_retgen",
        "query": query,
        "answer": output_answer,
        "retrieved": retrieved_documents,
        "llm_call": llm_calls,
        "iteration": num_iteration
    }