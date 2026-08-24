from components.llm import generate_answer
from components.retrieval import retrieve
from components.retrieval_data import initiate
from components.config import EMBEDDING_MODEL_NAME, TOP_K, TOP_N, LLM_MODEL, client
import json

def reciprocal_rank_fusion(retrieval_lists, k):
    fused = {}

    for retrieved_docs in retrieval_lists:
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc["id"]
            rrf_score = 1 / (k + rank)

            if doc_id not in fused:
                fused[doc_id] = {**doc, "rerank_score": 0.0}

            fused[doc_id]["rerank_score"] += rrf_score

    return sorted(fused.values(),key=lambda x: x["rerank_score"],reverse=True)

def generate_queries(query, num_sub_query, llm_model_name, client):
    prompt = f"""
                Bạn là một trợ lý tạo truy vấn tìm kiếm cho hệ thống RAG y tế bằng tiếng Việt.

                Nhiệm vụ:
                Tạo 4 truy vấn tìm kiếm khác nhau dựa trên câu hỏi của người dùng.
                Mỗi truy vấn phải diễn đạt một góc nhìn hoặc cách tìm kiếm khác nhau, nhưng vẫn giữ nguyên ý định và thông tin quan trọng của câu hỏi.

                Không trả lời câu hỏi.
                Không giải thích.
                Không thêm thông tin không có trong câu hỏi.

                Trả về CHỈ một JSON array gồm {num_sub_query} chuỗi truy vấn.

                Ví dụ:
                ["truy vấn 1", "truy vấn 2", "truy vấn 3", "truy vấn 4"]

                Câu hỏi:
                {query}
            """

    response = client.models.generate_content(
        model=llm_model_name,
        contents=prompt
    )

    sub_queries = json.loads(response.text)

    if not isinstance(sub_queries, list):
        raise ValueError("LLM did not return a list of queries.")

    sub_queries = [ q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]

    return sub_queries[:num_sub_query]

def rag_fusion_model(query, num_sub_query, evaluation_mode = False, k=60):
    corpus_documents, _, index, _ = initiate(query)

    sub_queries = generate_queries(query, num_sub_query, LLM_MODEL, client)

    retrieval_lists = []
    for i in range(num_sub_query):    
        retrieved = retrieve(EMBEDDING_MODEL_NAME, sub_queries[i], index, TOP_K, corpus_documents)
        print("RETRIEVED DOCUMENTS")

        retrieval_lists.append(retrieved)

    reranked = reciprocal_rank_fusion(retrieval_lists, k)
    reranked = reranked[:(TOP_N*num_sub_query)]
    print("Finished reranking")

    if evaluation_mode is False:
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

    if evaluation_mode is False:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print(answer)

    return {
        "query": query,
        "sub_queries": sub_queries,
        "retrieved": reranked,
        "answer": answer
    }

    
