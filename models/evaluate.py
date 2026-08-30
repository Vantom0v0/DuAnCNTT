from ragas.evaluation import evaluate
from ragas.metrics import _faithfulness, _answer_correctness, _context_precision, _context_recall
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.run_config import RunConfig

from components.config import async_client, EMBEDDING_MODEL_NAME, EVALUATOR_MODEL, LLM_MODEL

from datasets import Dataset
from dotenv import load_dotenv
import inspect
import json
import os
import pandas as pd

load_dotenv()

def create_data(model_result, s2g_mode):
    if not s2g_mode:
        return {
            "question": model_result["question"],
            "ground_truth": model_result["ground_truth"],
            "answer": model_result["answer"],
            "contexts": [doc["text"] for doc in model_result["retrieved"]]
        }

    contexts = []

    for iteration in model_result["retrieved"]:
        for doc in iteration:
            contexts.append(doc["text"])

    return {
        "question": model_result["question"],
        "ground_truth": model_result["ground_truth"],
        "answer": model_result["answer"],
        "contexts": contexts
    }

def load_ragas_data(filename):
    with open(filename, "r",  encoding="utf-8") as f:
        return json.load(f)

def re_format(data):
    return {
        "user_input": [chunk["question"] for chunk in data],
        "retrieved_contexts": [chunk["contexts"] for chunk in data],
        "response": [chunk["answer"] for chunk in data],
        "reference": [chunk["ground_truth"] for chunk in data]
    }

def ragas_eval(filename):
    ragas_data = load_ragas_data(filename)
    data = re_format(ragas_data)
    dataset = Dataset.from_dict(data)

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
    evaluator = llm_factory(model=LLM_MODEL, client=async_client)
    evaluator.model_args["max_completion_tokens"] = 4096
    evaluator.model_args.pop("max_tokens", None)
    evaluator_embeddings = embedding_factory(provider="huggingface", model=EMBEDDING_MODEL_NAME, client=async_client, interface="modern" )

    run_config = RunConfig(max_workers=4, max_retries=10, max_wait=60)

    result = evaluate(
        dataset=dataset, 
        metrics=[_faithfulness, _answer_correctness ,_context_precision ,_context_recall], 
        llm=evaluator, 
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    return result

def save_to_csv(result: Dataset, model_name):
    df = result.to_pandas()
    df.to_csv(f"evaluation/eval_results/{model_name}_eval_result.csv", index=False, encoding="utf-8-sig")

def generate_cost_data():
    for model_name in ["naive", "iter", "s2g", "fur"]:
        llm_calls = []
        iterations = []

        for idx in range(180):
            filename = f"generated_answers/{model_name}_answers/{model_name}_answer_{idx+1}.json"

            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            llm_calls.append(data.get("llm_call"))
            if model_name == "s2g":
                iterations.append(data.get("num_turns"))
            else:
                iterations.append(data.get("iteration"))

        df = pd.DataFrame({"llm_call": llm_calls, "iteration": iterations})

        df.to_csv(f"evaluation/eval_results/{model_name}_cost.csv", index=False)