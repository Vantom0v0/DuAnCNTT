from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pickle
import os

DOCS_DATA_PATH = "./data/documents.pkl"

def dataset_converter(data: Dataset, question_str: str, answer_str: str) -> list:
    """Convert a dataset into structured QA documents.
        Document structure:
        {
        "id": ,
        "text": f"Câu hỏi: {question}
                  Trả lời: {answer}",
        "question": question,
        "answer": answer
        }
    
    """
    
    documents = []

    for i, item in enumerate(data):
        question = (item.get(question_str) or "")
        answer = (item.get(answer_str) or "")

        question = str(question).strip()
        answer = str(answer).strip()

        if question and answer:
            documents.append({
                "id": i,
                "text": f"Câu hỏi: {question}\nTrả lời: {answer}",
                "question": question,
                "answer": answer
            })

    return documents

def construct_documents():
    """Construct and split the Vietnamese medical QA documents."""

    dataset1 = load_dataset("hungnm/vietnamese-medical-qa")
    dataset2 = load_dataset("urnus11/Vietnamese-Healthcare")

    data1 = dataset1["train"]
    data2 = dataset2["medical_qa"]

    documents1 = dataset_converter(data1, "question", "answer")
    document2 = dataset_converter(data2, "title", "content")
    documents = documents1 + document2

    corpus_documents, test_documents = train_test_split(documents, test_size=0.2, random_state=42)

    return corpus_documents, test_documents

def get_data():
    """Load cached data or construct it if it doesn't exist."""

    if os.path.exists(DOCS_DATA_PATH):
        print("Loading cached documents...")

        with open(DOCS_DATA_PATH, "rb") as f:
            corpus_documents, test_documents = pickle.load(f)

    else:
        corpus_documents, test_documents = construct_documents()
        
        print("Saving documents...")
        with open(DOCS_DATA_PATH, "wb") as f:
            pickle.dump(
                (corpus_documents, test_documents),
                f
            )

    return corpus_documents, test_documents