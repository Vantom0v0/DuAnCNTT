from components.preprocessing import get_data
from components.data_construct import get_index

def initiate():
    """Retrieve all necessary data for the retrieval"""

    corpus_documents, test_documents = get_data()

    texts = [document["text"] for document in corpus_documents]
    index = get_index(texts)

    return corpus_documents, test_documents, index, texts