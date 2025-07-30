from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
import requests
import zipfile

app = FastAPI()

# CORS: Allow frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torah-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/mnt/data")
ZIP_PATH = os.path.join(CHROMA_PATH, "chromadb.zip")
SQLITE_FILE = os.path.join(CHROMA_PATH, "chroma.sqlite3")
REMOTE_ZIP = "https://www.dropbox.com/scl/fi/019j9l2a58yb6489dkums/data.zip?rlkey=5w9arrpjbeqkdjpaay8qwlt7t&st=g7dr9ddt&dl=1"  # ✅ UPDATED to data.zip with dl=1

# Ensure directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Download and extract ZIP if needed
if not os.path.exists(SQLITE_FILE):
    print("⬇️ No existing database found — downloading from Dropbox...")
    response = requests.get(REMOTE_ZIP)

    print("📄 Content-Type:", response.headers.get("Content-Type"))
    print("🧪 First 100 bytes:", response.content[:100])

    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    if zipfile.is_zipfile(ZIP_PATH):
        print("✅ ZIP is valid. Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(CHROMA_PATH)
        print("📂 Extraction complete!")
        print("📁 Files in CHROMA_PATH:", os.listdir(CHROMA_PATH))  # ✅ Added logging
        os.remove(ZIP_PATH)
    else:
        raise ValueError("❌ The downloaded file is not a valid ZIP archive.")
else:
    print("✅ Existing database found — skipping download.")

# Initialize ChromaDB client
client = PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# Input schema
class QueryInput(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list[str]

@app.post("/query")
def query_torah_ai(input: QueryInput):
    results = []
    for source in input.sources:
        collection = client.get_or_create_collection(source, embedding_function=embedding_func)
        query_result = collection.query(
            query_texts=[input.prompt],
            n_results=3
        )
        results.append({source: query_result})
    return results
