import json
import re
from components.config import TOP_N, TOP_K, LLM_MODEL, client
from components.retrieval_data import initiate
from components.retrieval import retrieve
from components.llm import generate_answer
from components.reranker import rerank

def parse_json_response(text):
    """Parse JSON returned by the LLM."""

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)

def judge_evidence(query, evidence_context, max_gap_item):
    """
    Decide whether the current evidence is sufficient.
    If insufficient, identify structured information gaps.
    """

    if evidence_context:
        context = evidence_context
    else:
        context = "(Chưa có bằng chứng nào được truy xuất.)"

    prompt = f"""
                Bạn là bộ điều khiển retrieval cho một hệ thống hỏi đáp y tế bằng tiếng Việt.

                Nhiệm Vụ:

                Dựa CHỈ vào Câu Hỏi và Bằng Chứng Hiện Tại:
                1. Quyết định liệu bằng chứng hiện tại đã chứa đủ thông tin để trả lời câu hỏi một cách đáng tin cậy hay chưa.
                2. Nếu chưa đủ, xác định thông tin cụ thể còn thiếu để hệ thống có thể tìm kiếm ở lượt tiếp theo.

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

                Bằng Chứng Hiện Tại:
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

def build_query(query, gap_items, max_gap_item):
    """Convert structured gap items into the next retrieval query."""

    phrases = []
    for gap in gap_items[:max_gap_item]:
        target = gap.get("target", "").strip()
        slot = gap.get("slot", "").strip()
        description = gap.get("description", "").strip()

        if target and slot:
            phrase = f"{target} {slot}"
        elif description:
            phrase = description
        else:
            continue

        phrases.append(phrase)

    if not phrases:
        return query

    return query + " " + " ".join(phrases)

def split_sentences(text):
    """Simple Vietnamese sentence splitter."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def extract_evidence(query, gap_items, retrieved_documents, max_evidence_sentences):
    """
    Select useful sentences from newly retrieved documents.
    The LLM returns only the sentence indices.
    The actual sentences are then selected deterministically.
    """

    candidates = []
    for doc_idx, document in enumerate(retrieved_documents):
        sentences = split_sentences(document["text"])

        for sentence in sentences:
            candidates.append({"index": len(candidates), "document_id": document["id"], "text": sentence})

    if not candidates:
        return []

    candidate_text = "\n".join(f"[{candidate['index']}] {candidate['text']}" for candidate in candidates)
    gap_text = json.dumps(gap_items, ensure_ascii=False)

    prompt = f"""
                Bạn là bộ chọn bằng chứng cho hệ thống hỏi đáp y tế bằng tiếng Việt.

                Nhiệm vụ của bạn là chọn những câu quan trọng nhất từ danh sách các câu được truy xuất để giúp trả lời câu hỏi.

                Ưu tiên:
                1. Câu trực tiếp trả lời câu hỏi.
                2. Câu giải quyết thông tin còn thiếu trong gap items.
                3. Câu cung cấp thực thể trung gian hoặc mối quan hệ cần thiết.
                4. Câu giúp xác nhận hoặc loại bỏ sự mập mờ cho thông tin.

                KHÔNG được viết lại hoặc tạo câu mới.
                Chỉ trả về JSON array chứa các INDEX của những câu được chọn.
                Chọn tối đa {max_evidence_sentences} câu.

                Ví dụ:
                [0, 3, 7]

                CÂU HỎI:
                {query}

                GAP ITEMS:
                {gap_text}

                CÁC CÂU ĐƯỢC TRUY XUẤT:
                {candidate_text}
            """

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    selected_indices = parse_json_response(response.output_text)

    if not isinstance(selected_indices, list):
        return []

    selected_indices = [int(index) for index in selected_indices if isinstance(index, int) and 0 <= index < len(candidates)]
    selected_indices = selected_indices[:max_evidence_sentences]
    evidence = [candidates[index]["text"] for index in selected_indices]

    return evidence

def s2g_rag_model(query, groundtruth = "NA", evaluation_mode=False, max_iteration = 3, max_gap_item = TOP_N, max_evidence_sentences = 6):
    corpus_documents, _, index, _ = initiate()

    evidence_context = []
    retrieval_history = []
    llm_calls = 0
    num_iteration = 0
    for turn in range(max_iteration):
        print(f"\n{'=' * 70}")
        print(f"S2G-RAG TURN {turn + 1}")

        judge_result = judge_evidence(query, evidence_context, max_gap_item)
        sufficient = judge_result.get("sufficient", False)
        gap_items = judge_result.get("gap_items", [])
        llm_calls += 1

        print(f"Sufficient: {sufficient}")

        if not evaluation_mode:
            print("Gap items:")
            print(json.dumps(gap_items, ensure_ascii=False, indent=2))

        if sufficient:
            print("Evidence is sufficient. Stopping retrieval.")
            break

        retrieval_query = build_query(query, gap_items, max_gap_item)

        print(f"\nRetrieval query:")
        print(retrieval_query)

        retrieved_documents = retrieve(retrieval_query, index, TOP_K, corpus_documents)
        reranked_documents = rerank(query, retrieved_documents)
        retrieval_history.append(reranked_documents)

        if not evaluation_mode:
            for i, document in enumerate(reranked_documents):
                print(f"\n--- Document {i + 1} ---")
                print(f"Similarity: " f"{document['score']:.4f}")
                print(document["text"][:500])

        new_evidence = extract_evidence(query, gap_items, reranked_documents, max_evidence_sentences)
        llm_calls += 1

        for sentence in new_evidence:
            if sentence not in evidence_context:
                evidence_context.append(sentence)

        if not evaluation_mode:
            print("\nACCUMULATED EVIDENCE:")
            for i, sentence in enumerate(evidence_context):
                print(f"[{i + 1}] {sentence}")
        num_iteration += 1

    final_context = "\n".join(f"[Evidence {i + 1}] {sentence}" for i, sentence in enumerate(evidence_context))

    instruction = f"""
                    Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                    Hãy trả lời câu hỏi dựa CHỈ trên phần bằng chứng được cung cấp.
                    Không sử dụng kiến thức bên ngoài bằng chứng.
                    Nếu bằng chứng không đủ để trả lời, hãy nói rõ rằng không tìm thấy đủ thông tin.

                    Bằng chứng:
                    {final_context}
                """
    
    answer = generate_answer(
        query,
        [
            {
                "text": sentence,
                "id": i
            }
            for i, sentence in enumerate(evidence_context)
        ],
        LLM_MODEL,
        client,
        instruction
    )
    llm_calls += 1

    if not evaluation_mode:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print(answer)

    return {
        "model_name": "s2g_rag",
        "query": query,
        "groundtruth": groundtruth,
        "answer": answer,
        "evidence": evidence_context,
        "retrieval_history": retrieval_history,
        "llm_call": llm_calls,
        "num_turns": len(retrieval_history)
    }