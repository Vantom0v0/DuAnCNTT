from components.retrieval_data import initiate
from components.retrieval import retrieve
from components.reranker import rerank
from components.llm import generate_answer_raw
from components.config import EMBEDDING_MODEL_NAME, TOP_K, TOP_N, LLM_MODEL, client
from s2g_rag import parse_json_response
import numpy as np 
import json

def deduplicate_docs(new_docs, seen_ids):
    """
    Keep only documents whose IDs have not been retrieved previously.
    """
    unique_docs = []

    for doc in new_docs:
        doc_id = doc.get("id")

        if doc_id is None:
            unique_docs.append(doc)
            continue

        if doc_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc_id)

    return unique_docs, seen_ids

def format_evidence(evidence, queries):
    formatted = []

    for i, (query, docs) in enumerate(zip(queries, evidence)):
        formatted.append(
            f"Vòng lặp {i + 1}:\n"
            f"Câu hỏi: {query}\n"
            f"Bằng chứng:"
        )

        for doc in docs:
            formatted.append(f"- {doc['text']}")

    return "\n\n".join(formatted)

def docs_filter(local_reranked_documents, query, threshold, global_weight):
    global_reranked_documents = rerank(query, local_reranked_documents, len(local_reranked_documents), False)
    global_scores = [doc["rerank_score"] for doc in global_reranked_documents]
    local_scores = [doc["rerank_score"] for doc in local_reranked_documents]

    filtered = []
    for i, (global_score, local_score) in enumerate(zip(global_scores, local_scores)):
        weighted_sum_score = global_weight*global_score + (1-global_weight)*local_score
        if weighted_sum_score >= threshold:
            doc = local_reranked_documents[i].copy()
            doc["weighted_score"] = weighted_sum_score
            filtered.append(doc)

    filtered.sort(key=lambda x: x["weighted_score"], reverse=True)

    return filtered

def query_builder(gap_items, previous_queries):
    phrases = []
    for i, gap in enumerate(gap_items):
        target = gap.get("target", "").strip()
        slot = gap.get("slot", "").strip()
        description = gap.get("description", "").strip()

        if target and slot:
            phrase = f"Item {i+1}: {target} {slot}"
        elif description:
            phrase = f"Item {i+1}: {description}"
        else:
            continue

        phrases.append(phrase)

    gap_items_strings = "\n".join(phrases)

    prompt = f"""
                Bạn là một thành phần Query Builder trong hệ thống Retrieval-Augmented Generation (RAG) cho hỏi đáp y tế tiếng Việt.

                NHIỆM VỤ:
                Dựa CHỈ trên:
                1. Các gap items được cung cấp (là những thông tin, thực thể, mối quan hệ hoặc chi tiết còn thiếu/chưa rõ).
                2. Các câu hỏi đã được sử dụng trong những lần truy vấn trước.

                Bạn phải:
                1. Chọn DUY NHẤT một gap item quan trọng nhất và có khả năng cao nhất giúp giải quyết câu hỏi của người dùng.
                2. Tạo MỘT câu hỏi truy vấn tiếp theo (follow-up query) nhằm tìm kiếm thông tin cần thiết để giải quyết gap item đó.

                MỤC TIÊU CỦA FOLLOW-UP QUERY:
                - Trực tiếp tìm kiếm thông tin còn thiếu.
                - Cụ thể và có khả năng truy xuất tài liệu liên quan.
                - Tập trung vào gap item đã chọn.
                - Không hỏi lại những thông tin đã được tìm kiếm trong các câu hỏi trước.
                - Không mở rộng sang những vấn đề không liên quan đến câu hỏi ban đầu.
                - Không cố gắng trả lời câu hỏi; chỉ tạo câu hỏi dùng để RETRIEVE thông tin.

                QUY TẮC:
                - Chỉ sử dụng thông tin có trong gap items và previous queries.
                - KHÔNG sử dụng kiến thức bên ngoài để bổ sung hoặc suy đoán thông tin.
                - Không tạo ra thực thể, bệnh, thuốc, triệu chứng, quan hệ hoặc thông tin không xuất hiện trong input.
                - Nếu nội dung của một gap item đã được previous queries tìm kiếm, không ưu tiên gap item đó.
                - Nếu có nhiều gap items, ưu tiên gap item có ảnh hưởng lớn nhất đến khả năng trả lời câu hỏi ban đầu.
                - Follow-up query phải là một câu hỏi hoàn chỉnh bằng tiếng Việt.
                - Chỉ tạo DUY NHẤT một follow-up query.

                GAP ITEMS:
                {gap_items_strings}

                PREVIOUS QUERIES:
                {previous_queries[1:]}

                Câu Hỏi Của Người Dùng:
                {previous_queries[0]}

                OUTPUT FORMAT:
                - Chỉ trả về câu hỏi follow-up.
                - Không giải thích.
                - Không trả về gap item đã chọn.
                - Không đánh số.
            """
    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    return response.output_text

def judge(docs, query, max_gap_item):
    context_list = [f"Thông tin {i}:\n {doc['text']}" for i, doc in enumerate(docs)]
    context = "\n".join(context_list)

    prompt = f"""
                Bạn là một Judge retrieval cho một hệ thống hỏi đáp y tế bằng tiếng Việt.

                Nhiệm Vụ:

                Dựa CHỈ vào Câu Hỏi và Thông Tin Hiện Tại:
                1. Quyết định liệu những thông tin hiện tại đã chứa đủ thông tin để trả lời câu hỏi một cách đáng tin cậy hay chưa.
                2. Nếu chưa đủ, xác định thông tin cụ thể dựa trên câu hỏi và dữ liệu đã cung cấp còn thiếu để hệ thống có thể tìm kiếm ở lượt tiếp theo.

                QUAN TRỌNG:
                - Không sử dụng kiến thức bên ngoài bằng chứng.
                - Nếu bằng chứng không trực tiếp hỗ trợ câu trả lời, phải đặt "sufficient": false.
                - Nếu đủ bằng chứng để trả lời, đặt "sufficient": true.
                - Khi sufficient = true, "gap_items" PHẢI là [].
                - Khi sufficient = false, tạo tối đa {max_gap_item} gap item quan trọng nhất.

                Các loại gap:
                - "bridge_entity": thực thể trung gian còn thiếu
                - "attribute": thuộc tính còn thiếu
                - "relation": mối quan hệ còn thiếu
                - "evidence_span": bằng chứng cụ thể còn thiếu
                - "other": loại khác

                Câu Hỏi:
                {query}

                Thông Tin Hiện Tại:
                {context}

                Trả về chỉ một JSON object có chính xác dạng:

                {{
                    "sufficient": true hoặc false,
                    "gap_items": [
                        {{
                            "category": "bridge_entity | attribute | relation | evidence_span | other",
                            "target": "thực thể hoặc khái niệm đang được truy vấn",
                            "slot": "loại thông tin còn thiếu",
                            "description": "mô tả ngắn gọn thông tin còn thiếu"
                        }}
                    ]
                }}
            """

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    result = parse_json_response(response.output_text)
    return result

def adaptive_threshold(documents, c, stopping_check = False):
    if not documents:
        return None if stopping_check else (None, None)

    if stopping_check is False:
        scores = [doc["rerank_score"] for doc in documents]
        mean_value = np.mean(scores)
        std = np.std(scores)
        threshold_value = mean_value - c*std
        return threshold_value, mean_value, std
    else:
        scores = [doc["weighted_score"] for doc in documents]
        mean_value = np.mean(scores)
        return mean_value
        

def naive_retrieval(corpus_docs, index, query, return_amount):
    retrieved = retrieve(EMBEDDING_MODEL_NAME, query, index, TOP_K, corpus_docs)
    print("RETRIEVED DOCUMENTS")

    reranked = rerank(query, retrieved, return_amount)
    print("Finished reranking")

    return reranked

def fur(query, c = 0.95, max_iteration = 3, global_weight = 0.8, evaluation_mode = False):
    corpus_documents, _, index, _ = initiate(query)

    evidence = []
    queries = []

    reranked_docs = naive_retrieval(corpus_documents, index, query, TOP_N)
    threshold, mean_value, std = adaptive_threshold(reranked_docs, c)
    lowest_threshold = max(reranked_docs[-1]["rerank_score"], std)

    seen_ids = set()

    for doc in reranked_docs:
        if doc.get("id") is not None:
            seen_ids.add(doc["id"])

    if evaluation_mode is False:
        for i, doc in enumerate(reranked_docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Similarity: {doc['score']:.4f}")
            print(f"Rerank score: {doc['rerank_score']:.4f}")
            print(doc["text"][:500])

    evidence.append(reranked_docs)
    queries.append(query)

    num_iteration = 0
    num_llm_calls = 0
    for iteration in range(max_iteration):
        flat_evidence = [doc for round_docs in evidence for doc in round_docs]
        judge_decision = judge(flat_evidence, query, TOP_N)
        sufficient = judge_decision.get("sufficient", False)
        gap_items = judge_decision.get("gap_items", [])

        num_llm_calls += 1

        if sufficient == True:
            break

        if iteration == 0:
            sub_question = query_builder(gap_items, ["Chưa có câu hỏi follow-up nào."])
        else:
            sub_question = query_builder(gap_items, queries)
        queries.append(sub_question)
        num_llm_calls += 1

        reranked_docs = naive_retrieval(corpus_documents, index, sub_question, TOP_K)
        filtered_docs = docs_filter(reranked_docs, query, threshold, global_weight)
        filtered_docs, seen_ids = deduplicate_docs(filtered_docs, seen_ids)

        if evaluation_mode is False:
            for i, doc in enumerate(filtered_docs):
                print(f"\n--- Document {i+1} ---")
                print(f"Similarity: {doc['score']:.4f}")
                print(f"Rerank score: {doc['rerank_score']:.4f}")
                print(doc["text"][:500])

        num_iteration += 1

        if not filtered_docs:
            break

        evidence.append(filtered_docs)
        filtered_docs_mean = adaptive_threshold(filtered_docs, c, True)

        tolerence = 0.75
        expansion_threshold = mean_value - tolerence*lowest_threshold
        if filtered_docs_mean < expansion_threshold:
            break


    final_evidence = format_evidence(evidence, queries)

    instruction = f"""
                    Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                    Hãy trả lời câu hỏi dựa CHỈ trên phần thông tin được cung cấp.
                    Không sử dụng kiến thức bên ngoài bằng chứng.
                    Nếu bằng chứng không đủ để trả lời, hãy nói rõ rằng không tìm thấy đủ thông tin.

                    Thông tin tìm được sau từng vòng lặp follow-up question:
                    {final_evidence}
                """
    
    answer = generate_answer_raw(LLM_MODEL, client, instruction)
    num_llm_calls += 1

    if not evaluation_mode:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print(answer)

    retrieved_documents = [doc for round_docs in evidence for doc in round_docs]

    return {
        "model_name": "fur",
        "query": query,
        "answer": answer,
        "evidence": retrieved_documents,
        "llm_call": num_llm_calls,
        "iteration": num_iteration
    }