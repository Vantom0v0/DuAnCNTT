from naive_rag import naive_rag_model
from iter_retgen import iter_retgen_model
from ircot import ircot_model
from s2g_rag import s2g_rag_model
from follow_up_retrieval import fur

import json
import os

from components.preprocessing import get_data


def save(file_name, answer):
    directory = os.path.dirname(file_name)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)


def models(num, query, question_number, evaluation_mode):
    match num:

        case 0:
            # Naive RAG
            naive_answer = naive_rag_model(query, evaluation_mode=evaluation_mode)

            naive_answer_file = (
                f"generated_answers/naive_answers/"
                f"naive_answer_{question_number}.json"
            )

            save(naive_answer_file, naive_answer)

        case 1:
            # Iter-RetGen
            iteration = 3

            iter_answer = iter_retgen_model(query, iteration, evaluation_mode=evaluation_mode)

            iter_answer_file = (
                f"generated_answers/iter_answers/"
                f"iter_answer_{question_number}.json"
            )

            save(iter_answer_file, iter_answer)

        case 2:
            # S2G-RAG
            s2g_answer = s2g_rag_model(query, evaluation_mode=evaluation_mode)

            s2g_answer_file = (
                f"generated_answers/s2g_answers/"
                f"s2g_answer_{question_number}.json"
            )

            save(s2g_answer_file, s2g_answer)

        case 3:
            # FUR
            fur_answer = fur(query, evaluation_mode=evaluation_mode)

            fur_answer_file = (
                f"generated_answers/fur_answers/"
                f"fur_answer_{question_number}.json"
            )

            save(fur_answer_file, fur_answer)

        case _:
            return


if __name__ == "__main__":

    _, test_documents = get_data()
    
    # Example: run question 1
    rand_list = [96, 523, 372, 863, 28]
    question_number = rand_list[0]

    query = test_documents[question_number - 1]["question"]
    answer = test_documents[question_number - 1]["answer"]

    # 0: naive, 1: iter, 2: s2g, 3: fur
    # for i in range(4):
    models(3, query, question_number, True)

    print("---------------------------------------------------------")
    print("Question:", query)
    print("Answer:", answer)