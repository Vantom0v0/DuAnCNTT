import os
from dotenv import load_dotenv
# from google import genai
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# LLM_MODEL = "gemini-3.5-flash"
LLM_MODEL = "gpt-5.6-luna"

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

RERANK_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3")

TOP_K = 10

TOP_N = 5