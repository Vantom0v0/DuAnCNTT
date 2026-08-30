from datasets import load_dataset, Dataset 
import random 
import pickle 
import os 

DOCS_DATA_PATH = "./data/documents_full.pkl" 
def dataset_converter(data: Dataset, question_str: str, answer_str: str, src: str) -> list: 
    """
    Convert a dataset into structured QA documents. 
    Document structure: { 
        "id": , 
        "text": f"Câu hỏi: {question} Trả lời: {answer}", 
        "question": question, 
        "answer": answer 
    } 
    """ 
    documents = [] 

    for i, item in enumerate(data): 
        question = str((item.get(question_str) or "")).strip() 
        answer = str(item.get(answer_str) or "").strip() 

        if question and answer: 
            documents.append({ 
                "id": f"{src}_{i}", 
                "source": src, 
                "text": f"Câu hỏi: {question}\nTrả lời: {answer}", 
                "question": question, 
                "answer": answer }) 

    return documents 

def construct_documents(): 
    """Construct and split the Vietnamese medical QA documents.""" 

    dataset1 = load_dataset("hungnm/vietnamese-medical-qa") 
    dataset2 = load_dataset("urnus11/Vietnamese-Healthcare") 
    dataset3 = load_dataset("Dqdung205/medical-vietnamese-qa") 

    data1 = dataset1["train"] 
    data2_1 = dataset2["medical_qa"] 
    data2_2 = dataset2["vinmec_article_subtitle"] 
    data3 = dataset3["train"] 

    documents1 = dataset_converter(data1, "question", "answer", "vietnamese_medical_qa") 
    document2_1 = dataset_converter(data2_1, "title", "content", "vietnamese_healthcare") 
    document2_2 = dataset_converter(data2_2, "title", "content", "vietnamese_healthcare") 
    document3 = dataset_converter(data3, "question", "answer", "medical_vietnamese_qa") 

    documents = documents1 + document2_1 + document2_2 + document3 

    random.seed(42) 

    test_document1 = random.sample(document2_1, 60) 
    test_document2 = random.sample(documents1, 60) 
    test_document3 = random.sample(document3, 60) 
    test_documents = test_document1 + test_document2 + test_document3 

    test_ids = {doc["id"] for doc in test_documents} 
    corpus_documents = [doc for doc in documents if doc["id"] not in test_ids] 

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