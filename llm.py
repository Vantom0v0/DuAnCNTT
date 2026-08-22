def generate_answer(query, retrieved_documents, llm_model_name, client, instruction):

    context = "\n\n".join(
        [
            f"[Tài liệu {i+1}]\n{doc['text']}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    prompt = f"""
                {instruction}

                Câu hỏi:
                {query}

                Tài liệu tham khảo:
                {context}

                Hãy đưa ra câu trả lời bằng tiếng Việt rõ ràng, liền mạch và xuyên suốt trong một đoạn văn ngắn gọn.
            """

    response = client.models.generate_content(
        model=llm_model_name,
        contents=prompt
    )

    return response.text