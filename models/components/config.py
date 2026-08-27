import os
from dotenv import load_dotenv
# from google import genai
from openai import OpenAI
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# LLM_MODEL = "gemini-3.5-flash"
LLM_MODEL = "gpt-5.6-luna"

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

RERANK_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3")

TOP_K = 10
TOP_N = 5

DOCS_DATA_PATH = "./data/documents_full.pkl"
INDEX_PATH = "./vector_storage/medical_full.index"

HEART_KEYWORDS = [
    # Cardiovascular / general
    "tim mạch",
    "tim",
    "mạch vành",
    "bệnh mạch vành",
    "động mạch vành",
    "cao huyết áp",
    "tăng huyết áp",
    "huyết áp",
    "hạ huyết áp",
    "suy tim",
    "nhồi máu cơ tim",
    "đau thắt ngực",
    "thiếu máu cơ tim",
    "rối loạn nhịp tim",
    "loạn nhịp tim",
    "rung nhĩ",
    "ngoại tâm thu",
    "van tim",
    "bệnh van tim",
    "xơ vữa động mạch",
    "cholesterol",
    "triglyceride",
    "mỡ máu",
    "lipid máu",
    "đột quỵ",
    "tai biến mạch máu não",
    "mạch máu",

    # Symptoms
    "đau ngực",
    "khó thở",
    "hồi hộp",
    "đánh trống ngực",
    "ngất",

    # Diagnosis / procedures
    "điện tâm đồ",
    "siêu âm tim",
    "thông tim",
    "đặt stent",
    "bắc cầu động mạch vành",
    "máy tạo nhịp tim",

    # Cardiovascular diseases
    "bệnh cơ tim",
    "viêm cơ tim",
    "viêm màng ngoài tim",
    "viêm nội tâm mạc",
    "phình động mạch",
    "huyết khối",
    "nhịp tim nhanh",
    "nhịp tim chậm",
    "ngừng tim",
    "ngưng tim",

    # Cardiovascular treatment
    "chống đông",
]


DIABETES_KEYWORDS = [
    # Diabetes / glucose
    "tiểu đường",
    "đái tháo đường",
    "bệnh tiểu đường",
    "bệnh đái tháo đường",
    "đường huyết",
    "đường máu",
    "glucose",
    "hba1c",
    "HbA1c",
    "hạ đường huyết",
    "tăng đường huyết",

    # Diabetes types / conditions
    "đái tháo đường type 1",
    "đái tháo đường type 2",
    "đái tháo đường tuýp 1",
    "đái tháo đường tuýp 2",
    "tiền đái tháo đường",
    "rối loạn dung nạp glucose",
    "tiểu đường thai kỳ",
    "đái tháo đường thai kỳ",
    "kháng insulin",

    # Treatment
    "insulin",
    "metformin",
    "thuốc hạ đường huyết",

    # Complications
    "biến chứng tiểu đường",
    "biến chứng đái tháo đường",
    "bệnh võng mạc đái tháo đường",
    "bệnh thận đái tháo đường",
    "bệnh thận do đái tháo đường",
    "biến chứng thần kinh",
    "bệnh thần kinh đái tháo đường",
    "bệnh lý thần kinh do đái tháo đường",
    "bàn chân đái tháo đường",
    "loét bàn chân",

    # Acute complication
    "nhiễm toan ceton",
]