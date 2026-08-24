from components.preprocessing import get_data
from components.data_construct import get_index
from components.retrieval import retrieve
from components.config import EMBEDDING_MODEL_NAME

def initiate(query):
    """Retrieve all necessary data for the retrieval"""

    corpus_documents, test_documents = get_data()

    texts = [document["text"] for document in corpus_documents]
    index = get_index(EMBEDDING_MODEL_NAME, texts)

    return corpus_documents, test_documents, index, texts