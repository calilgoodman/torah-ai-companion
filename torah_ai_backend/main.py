from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
import requests
import zipfile

app = FastAPI()

# CORS setup
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
REMOTE_ZIP = "https://www.dropbox.com/scl/fi/62cl7p8k1neato926iqnc/chromadb_minimal.zip?rlkey=m91n3uznotbo2li4ra8qinb8i&st=0ntk8xpv&dl=1"

# Ensure ChromaDB folder exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Download & extract ChromaDB zip if not already present
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
    else:
        raise ValueError("❌ The downloaded file is not a valid ZIP archive.")

    os.remove(ZIP_PATH)
else:
    print("✅ Existing database found — skipping download.")

# Initialize ChromaDB client
client = PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# Input model
class QueryInput(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list[str]

# Query endpoint
@app.post("/query")
def query_torah_ai(input: QueryInput):
    results = []
    for source in input.sources:
        collection = client.get_or_create_collection(source, embedding_function=embedding_func)

        query_result = collection.query(
            query_texts=[input.prompt],
            n_results=3,
            include=["documents", "metadatas"]
        )

        results.append({source: query_result})

    return results
