import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

LLM_MODEL = "gemini-3.5-flash"

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

TOP_K = 10

TOP_N = 5