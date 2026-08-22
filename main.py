from naive_rag import naive_rag_model

if __name__ == "__main__":
    query = "Em mới mổ xương đòn được 1 năm 4 tháng, em đang có chút công việc nên chưa mổ rút nẹp được. Vậy bác sĩ cho em hỏicó thể mổ rút nẹp xương đòn sau 2 năm được không?Em cảm ơn bác sĩ."

    naive_answer = naive_rag_model(query)

