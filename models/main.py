from naive_rag import naive_rag_model
from iter_retgen import iter_retgen_model
from s2g_rag import s2g_rag_model
from follow_up_retrieval import fur

import json
import os

from components.preprocessing import get_data
from components.retrieval_data import initiate
from evaluate import create_data, ragas_eval, save_to_csv, generate_cost_data

def save(file_name, content):
    directory = os.path.dirname(file_name)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def models(num, query, groundtruth, question_number, evaluation_mode):
    s2g_mode = False

    match num:

        case 0:
            # Naive RAG
            result = naive_rag_model(query, groundtruth, evaluation_mode=evaluation_mode)
            name = result["model_name"]

            output_file = (
                f"generated_answers/naive_answers/"
                f"naive_answer_{question_number}.json"
            )

        case 1:
            # Iter-RetGen
            iteration = 3

            result = iter_retgen_model(query, groundtruth, iteration, evaluation_mode=evaluation_mode)
            name = result["model_name"]

            output_file = (
                f"generated_answers/iter_answers/"
                f"iter_answer_{question_number}.json"
            )

        case 2:
            # S2G-RAG
            s2g_mode = True
            result = s2g_rag_model(query, groundtruth, evaluation_mode=evaluation_mode)
            name = result["model_name"]

            output_file = (
                f"generated_answers/s2g_answers/"
                f"s2g_answer_{question_number}.json"
            )

        case 3:
            # FUR
            result = fur(query, groundtruth, evaluation_mode=evaluation_mode)
            name = result["model_name"]

            output_file = (
                f"generated_answers/fur_answers/"
                f"fur_answer_{question_number}.json"
            )

        case _:
            raise ValueError(f"Unknown model index: {num}")

    save(output_file, result)

    return create_data(result, s2g_mode), name

def generate_eval_data(test_documents, model_amounts: int, question_amounts: int):
    for model_idx in range(model_amounts):
        all_eval_data = []

        for question_number in range(question_amounts):
            query = test_documents[question_number]["question"]
            answer = test_documents[question_number]["answer"]

            # 0: naive, 1: iter, 2: s2g, 3: fur
            eval_data, model_name = models(model_idx, query, answer, question_number + 1, True)
            all_eval_data.append(eval_data)

            eval_data_file_name = (f"evaluation/eval_data/{model_name}.json")

            save(eval_data_file_name, all_eval_data)

            print(
                f"{model_name}: "
                f"{question_number + 1}/{len(test_documents)} completed"
            )
            
        print(f"Finished {model_name}")

def generate_eval_results(model_idx: int):
    model_name = ["naive_rag", "iter_retgen", "s2g_rag", "fur"]
    result = ragas_eval(f"evaluation/eval_data/{model_name[model_idx]}.json")
    save_to_csv(result, model_name[model_idx])

if __name__ == "__main__":
    _, test_documents = get_data()
    # assert len(test_documents) == 180, (f"Expected 180 questions, got {len(test_documents)}")
    # save("data/test_documents.json", test_documents)

    # generate_eval_data(test_documents, model_amounts=4, question_amounts=len(test_documents))

    # 0: naive, 1: iter, 2: s2g, 3: fur
    # for model_idx in range(4):
    # generate_eval_results(3)

    generate_cost_data()
