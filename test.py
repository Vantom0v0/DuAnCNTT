import os

INDEX_PATH = "./vector_storage/medical.index"

if os.path.exists(INDEX_PATH):
    print(f"Retrieving vector index at {INDEX_PATH}")