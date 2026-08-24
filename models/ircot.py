import re
from components.config import EMBEDDING_MODEL_NAME, TOP_N, LLM_MODEL, client
from components.retrieval_data import initiate
from components.retrieval import retrieve
from components.llm import generate_answer

ANSWER_MARKER = "câu trả lời là"

DEMONSTRATIONS = [
    {
        "paragraphs": [
            {
                "text": "Tăng huyết áp là tình trạng huyết áp động mạch tăng cao kéo dài. Bệnh thường không gây ra triệu chứng rõ ràng trong giai đoạn đầu."
            },
            {
                "text": "Tăng huyết áp kéo dài có thể làm tăng nguy cơ đột quỵ, bệnh tim mạch, suy tim và tổn thương thận."
            },
        ],
        "question": "Tại sao tăng huyết áp được xem là yếu tố nguy cơ của đột quỵ?",
        "cot": (
            "Tăng huyết áp có thể không gây triệu chứng rõ ràng nhưng làm tăng nguy cơ nhiều biến chứng tim mạch. "
            "Tăng huyết áp kéo dài làm tăng nguy cơ đột quỵ. "
            f"Do đó {ANSWER_MARKER}: tăng huyết áp kéo dài làm tăng nguy cơ đột quỵ."
        ),
    },

    {
        "paragraphs": [
            {
                "text": "Đái tháo đường type 2 xảy ra khi cơ thể sử dụng insulin không hiệu quả và có thể dần giảm khả năng sản xuất insulin."
            },
            {
                "text": "Kiểm soát đường huyết kém trong thời gian dài có thể gây tổn thương các mạch máu và dây thần kinh."
            },
            {
                "text": "Tổn thương thần kinh do đái tháo đường có thể gây tê, đau hoặc mất cảm giác ở bàn chân."
            },
        ],
        "question": "Tại sao người mắc đái tháo đường lâu năm có thể bị tê chân?",
        "cot": (
            "Đái tháo đường type 2 có thể dẫn đến kiểm soát đường huyết kém trong thời gian dài. "
            "Đường huyết được kiểm soát kém có thể gây tổn thương dây thần kinh. "
            "Tổn thương dây thần kinh có thể gây tê hoặc mất cảm giác ở bàn chân. "
            f"Do đó {ANSWER_MARKER}: tổn thương thần kinh do đái tháo đường có thể gây tê chân."
        ),
    },

    {
        "paragraphs": [
            {
                "text": "Thiếu máu thiếu sắt xảy ra khi cơ thể không có đủ sắt để tạo hemoglobin."
            },
            {
                "text": "Hemoglobin là protein trong hồng cầu có chức năng vận chuyển oxy đến các mô trong cơ thể."
            },
            {
                "text": "Thiếu máu có thể gây mệt mỏi, yếu sức, chóng mặt và da nhợt nhạt."
            },
        ],
        "question": "Vì sao thiếu máu thiếu sắt có thể khiến người bệnh mệt mỏi?",
        "cot": (
            "Thiếu sắt làm cơ thể không có đủ nguyên liệu để tạo hemoglobin. "
            "Hemoglobin có chức năng vận chuyển oxy đến các mô. "
            "Khi khả năng vận chuyển oxy bị ảnh hưởng, người bệnh có thể cảm thấy mệt mỏi. "
            f"Do đó {ANSWER_MARKER}: thiếu máu thiếu sắt có thể gây mệt mỏi do ảnh hưởng đến việc vận chuyển oxy."
        ),
    },

    {
        "paragraphs": [
            {
                "text": "Béo phì được xác định dựa trên chỉ số khối cơ thể (BMI), trong đó BMI từ 30 trở lên ở người trưởng thành thường được phân loại là béo phì."
            },
            {
                "text": "Béo phì có liên quan đến tình trạng kháng insulin, trong đó tế bào đáp ứng kém với insulin."
            },
            {
                "text": "Kháng insulin làm cơ thể khó duy trì mức đường huyết bình thường và có thể góp phần vào sự phát triển của đái tháo đường type 2."
            },
        ],
        "question": "Béo phì có liên quan như thế nào đến nguy cơ mắc đái tháo đường type 2?",
        "cot": (
            "Béo phì có liên quan đến tình trạng kháng insulin. "
            "Kháng insulin khiến tế bào đáp ứng kém với insulin và làm cơ thể khó duy trì đường huyết bình thường. "
            f"Do đó {ANSWER_MARKER}: béo phì có thể làm tăng nguy cơ đái tháo đường type 2 thông qua kháng insulin."
        ),
    },
]

def render_demo(demo: dict) -> str:
    """Restructure DEMONSTRATION into a clean format"""

    para_block = "\n\n".join(f"{p['text']}" for p in demo["paragraphs"])
    return f"{para_block}\n\nQ: {demo['question']}\nA: {demo['cot']}"

def first_sentence_retriever(text: str) -> str:
    """Keep only the first generated sentence per reason-step"""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()

def dedupe_merge(collected: dict, new_docs: list) -> dict:
    """Check for duplication then merge new documents"""
    
    for doc in new_docs:
        key = doc["text"][:120]
        if key not in collected:
            collected[key] = doc
    return collected

def ircot_model(query, max_steps=8, max_paragraphs=15, evaluation_mode=False):
    corpus_documents, _, vector_index, _ = initiate(query)
    demo_block = "\n\n".join(render_demo(d) for d in DEMONSTRATIONS)
    collected = {}
    cot_sentences = []
    sentence = query

    for step_idx in range(max_steps):
        print(f"\nIteration {step_idx + 1}:")

        retrieved_docs = retrieve(EMBEDDING_MODEL_NAME, sentence, vector_index, TOP_N, corpus_documents)
        collected = dedupe_merge(collected, retrieved_docs)

        if evaluation_mode is False:
            print(f"Retrieved {len(retrieved_docs)} new paragraphs using: '{sentence}'")
            for doc in retrieved_docs:
                print(f"Similarity: {doc['score']:.4f}")
                print(doc["text"][:500])

        cot_prefix = " ".join(cot_sentences)

        instruction = f"""
                        Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                        Dưới đây là một số ví dụ minh hoạ cách suy luận từng bước và kết thúc bằng cụm từ "{ANSWER_MARKER}":

                        {demo_block}

                        Bây giờ, chỉ dựa trên các đoạn tài liệu được cung cấp, hãy viết TIẾP một câu suy luận duy nhất
                        (không lặp lại các câu đã có, không trả lời nhiều câu cùng lúc) để tiếp nối phần suy luận sau đây.
                        Nếu đã đủ thông tin để trả lời, hãy kết thúc câu bằng cụm từ "{ANSWER_MARKER}: <câu trả lời>".

                        Suy luận đã có: {cot_prefix if cot_prefix else "(chưa có)"}
                    """

        raw = generate_answer(query, list(collected.values())[:max_paragraphs], LLM_MODEL, client, instruction)
        sentence = first_sentence_retriever(raw)
        cot_sentences.append(sentence)

        if evaluation_mode is False:
            print(f"CoT sentence {step_idx + 1}: {sentence}")

        if ANSWER_MARKER in sentence.lower():
            break

    final_paragraphs = list(collected.values())[:max_paragraphs]

    qa_instruction = f"""
                        Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.
                        Hãy trả lời câu hỏi của người dùng dựa trên suy luận trước đó và thông tin trong phần tài liệu được cung cấp.
                        Suy luận trước đó: {" ".join(cot_sentences)}
                    """
    final_answer = generate_answer(query, final_paragraphs, LLM_MODEL, client, qa_instruction)
    if evaluation_mode is False:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print(final_answer)

    return {
        "query": query,
        "cot": cot_sentences,
        "answer": final_answer,
        "retrieved": final_paragraphs,
        "num_steps": len(cot_sentences),
    }