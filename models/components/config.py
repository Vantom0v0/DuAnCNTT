import os
from dotenv import load_dotenv
# from google import genai
from openai import OpenAI
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))

# LLM_MODEL = "gemini-3.5-flash"
LLM_MODEL = "gpt-5.6-luna"
EVALUATOR_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

RERANK_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3")

TOP_K = 10
TOP_N = 5

DOCS_DATA_PATH = "./data/documents_full.pkl"
INDEX_PATH = "./vector_storage/medical_full.index"