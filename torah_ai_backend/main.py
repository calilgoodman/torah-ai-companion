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
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/mnt/data/chromadb")
ZIP_PATH = os.path.join(CHROMA_PATH, "chromadb.zip")
REMOTE_ZIP = "https://www.dropbox.com/scl/fi/aosp4l255q7osp2ofipkg/chromadb_backup.zip?rlkey=xi544ppm8lodehif01199leq8&st=1dewihwv&dl=1"
COLLECTIONS_PATH = os.path.join(CHROMA_PATH, "collections")

# Ensure ChromaDB directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Download & extract backup zip if full ChromaDB is missing
if not os.path.exists(COLLECTIONS_PATH):
    print("⬇️ No ChromaDB collections found — downloading backup from Dropbox...")
    response = requests.get(REMOTE_ZIP)

    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    if zipfile.is_zipfile(ZIP_PATH):
        print("✅ ZIP is valid. Extracting contents...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(CHROMA_PATH)
        print(f"📂 Extraction complete to {CHROMA_PATH}")
    else:
        raise ValueError("❌ The downloaded file is not a valid ZIP archive.")

    os.remove(ZIP_PATH)
else:
    print("✅ ChromaDB already initialized — skipping download.")

# Initialize Chroma client
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
