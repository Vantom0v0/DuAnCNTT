from datasets import load_dataset, Dataset
from components.config import HEART_KEYWORDS, DIABETES_KEYWORDS, DOCS_DATA_PATH
import random
import pickle
import os

def classify_document(text):
    """
    Determine whether a document belongs to Heart or Diabetes.
    Returns:
        "heart"
        "diabetes"
        "both"
        None
    """

    text_lower = text.lower()
    heart = any(keyword.lower() in text_lower for keyword in HEART_KEYWORDS)
    diabetes = any(keyword.lower() in text_lower for keyword in DIABETES_KEYWORDS)

    if heart and diabetes:
        return "both"

    if heart:
        return "heart"

    if diabetes:
        return "diabetes"

    return None

def dataset_converter(data: Dataset, question_str: str, answer_str: str, src: str) -> list:
    documents = []

    for i, item in enumerate(data):
        question = str(item.get(question_str) or "").strip()
        answer = str(item.get(answer_str) or "").strip()

        if not question or not answer:
            continue

        text = f"Câu hỏi: {question}\nTrả lời: {answer}"
        topic = classify_document(text)

        if topic is None:
            continue

        documents.append({
            "id": f"{src}_{i}",
            "source": src,
            "topic": topic,
            "text": text,
            "question": question,
            "answer": answer
        })

    return documents

def construct_documents():
    dataset1 = load_dataset("hungnm/vietnamese-medical-qa")
    dataset2 = load_dataset("urnus11/Vietnamese-Healthcare")
    dataset3 = load_dataset("Dqdung205/medical-vietnamese-qa")

    data1 = dataset1["train"]
    data2_1 = dataset2["medical_qa"]
    data2_2 = dataset2["vinmec_article_subtitle"]
    data3 = dataset3["train"]

    documents1 = dataset_converter(data1, "question", "answer", "vietnamese_medical_qa")
    documents2_1 = dataset_converter(data2_1, "title", "content", "vietnamese_healthcare")
    documents2_2 = dataset_converter(data2_2, "title", "content", "vietnamese_healthcare")
    documents3 = dataset_converter(data3, "question", "answer", "medical_vietnamese_qa")

    documents = (documents1 + documents2_1 + documents2_2 + documents3)

    print(f"Total Heart + Diabetes documents: {len(documents)}")
    heart_count = sum(1 for doc in documents if doc["topic"] in ["heart", "both"])
    diabetes_count = sum(1 for doc in documents if doc["topic"] in ["diabetes", "both"])
    both_count = sum(1 for doc in documents if doc["topic"] == "both")

    print(f"Heart:     {heart_count}")
    print(f"Diabetes:  {diabetes_count}")
    print(f"Both:      {both_count}")

    heart_documents = [doc for doc in documents if doc["topic"] == "heart"]
    diabetes_documents = [doc for doc in documents if doc["topic"] == "diabetes"]

    random.seed(42)
    test_heart = random.sample(heart_documents, min(100, len(heart_documents)))
    test_diabetes = random.sample(diabetes_documents, min(100, len(diabetes_documents)))
    test_documents = test_heart + test_diabetes

    test_ids = {doc["id"] for doc in test_documents}

    corpus_documents = [doc for doc in documents if doc["id"] not in test_ids]

    print(f"Total test questions: {len(test_documents)}")
    print(f"Heart test questions: {len(test_heart)}")
    print(f"Diabetes test questions: {len(test_diabetes)}")
    print(f"Corpus documents: {len(corpus_documents)}")

    return corpus_documents, test_documents

def get_data():
    if os.path.exists(DOCS_DATA_PATH):
        print("Loading cached documents...")

        with open(DOCS_DATA_PATH, "rb") as f:
            corpus_documents, test_documents = pickle.load(f)
    else:
        corpus_documents, test_documents = construct_documents()
        print("Saving documents...")

        with open(DOCS_DATA_PATH, "wb") as f:
            pickle.dump((corpus_documents, test_documents), f)

    return corpus_documents, test_documents