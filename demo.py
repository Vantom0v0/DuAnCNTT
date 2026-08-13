# ============================================================
# Vietnamese Healthcare - Naive RAG Demo
# ============================================================
#
# Pipeline:
# Vietnamese dataset
#       ↓
# Chunk documents
#       ↓
# Vietnamese embeddings
#       ↓
# FAISS vector store
#       ↓
# Retrieve top-k chunks
#       ↓
# Gemini generates answer from retrieved context
#
# ============================================================

# If needed, install these first:
# pip install datasets sentence-transformers faiss-cpu google-genai

import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
from google import genai

load_dotenv()

# CONFIGURATION
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
LLM_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
TOP_K = 5


# LOAD DATASET
print("Loading Vietnamese healthcare dataset...")
dataset = load_dataset("hungnm/vietnamese-medical-qa")

# CONVERT DATASET INTO TEXT DOCUMENTS
documents = []

data = dataset["train"]

for i, item in enumerate(data):
    question = (item.get("question") or "")
    answer = (item.get("answer") or "")

    question = str(question).strip()
    answer = str(answer).strip()

    if question and answer:
        documents.append({
            "id": i,
            "text": f"Câu hỏi: {question}\nTrả lời: {answer}",
            "question": question,
            "answer": answer
        })

print("\nUsable documents:", len(documents))


# SIMPLE CHUNKING
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
chunks = []

for doc in documents[:100]:
    text = doc["text"]
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "source_id": doc["id"]
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

print("Number of chunks:", len(chunks))


# LOAD VIETNAMESE EMBEDDING MODEL
print("\nLoading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")


# CREATE EMBEDDINGS
print("\nCreating embeddings...")
texts = [chunk["text"] for chunk in chunks]
embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# BUILD FAISS VECTOR INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(
    embeddings.astype("float32")
)

print("\nFAISS index created.")
print("Indexed vectors:", index.ntotal)


# RETRIEVAL FUNCTION

def retrieve(query, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "score": float(score),
            "source_id": chunks[idx]["source_id"]
        })

    return results


# GENERATION FUNCTION
def generate_answer(query, retrieved_documents):

    context = "\n\n".join(
        [
            f"[Tài liệu {i+1}]\n{doc['text']}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )

    prompt = f"""
                Bạn là một trợ lý hỏi đáp y tế bằng tiếng Việt.

                Hãy trả lời câu hỏi của người dùng CHỈ dựa trên thông tin trong phần tài liệu được cung cấp.

                Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ rằng không tìm thấy đủ thông tin thay vì tự bịa thêm kiến thức.

                Không sử dụng kiến thức bên ngoài tài liệu.

                Câu hỏi:
                {query}

                Tài liệu tham khảo:
                {context}

                Hãy đưa ra câu trả lời rõ ràng, ngắn gọn bằng tiếng Việt.
            """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text


# RUN NAIVE RAG
def naive_rag(query):
    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)
    print(query)

    retrieved = retrieve(query)
    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for i, doc in enumerate(retrieved):

        print(f"\n--- Document {i+1} ---")
        print(f"Similarity: {doc['score']:.4f}")
        print(doc["text"][:500])

    answer = generate_answer(
        query,
        retrieved
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    return {
        "query": query,
        "retrieved": retrieved,
        "answer": answer
    }


# TEST QUESTION
result = naive_rag(
    "Những triệu chứng phổ biến của bệnh tiểu đường là gì?"
)