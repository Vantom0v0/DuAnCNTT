from preprocessing import get_data
from data_construct import get_index
from retrieval import retrieve
from config import EMBEDDING_MODEL_NAME

def initiate(query):
    """Retrieve all necessary data for the first retrieval"""

    print("Loading Vietnamese healthcare dataset...")
    corpus_documents, test_documents = get_data()
    print("Documents loaded.")

    print("Getting vector index")
    texts = [document["text"] for document in corpus_documents]
    index = get_index(EMBEDDING_MODEL_NAME, texts)
    print("Index accquired.")

    return corpus_documents, test_documents, index, texts