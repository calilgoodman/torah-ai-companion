from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from torah_ai_backend.query_rewriter import generate_semantic_query
import os
import requests
import zipfile

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torahlifeguide.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and Constants
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/mnt/data")
ZIP_PATH = os.path.join(CHROMA_PATH, "chromadb.zip")
SQLITE_FILE = os.path.join(CHROMA_PATH, "chroma.sqlite3")
REMOTE_ZIP = "https://www.dropbox.com/scl/fi/019j9l2a58yb6489dkums/data.zip?rlkey=5w9arrpjbeqkdjpaay8qwlt7t&st=g7dr9ddt&dl=1"

# Ensure directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Download ChromaDB ZIP if it doesn't exist
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
        print("📁 Files in CHROMA_PATH:", os.listdir(CHROMA_PATH))
        os.remove(ZIP_PATH)
    else:
        raise ValueError("❌ The downloaded file is not a valid ZIP archive.")
else:
    print("✅ Existing database found — skipping download.")

# Initialize ChromaDB client
client = PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# Request schema
class QueryInput(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list[str]

@app.post("/query")
def query_torah_ai(input: QueryInput):
    results = []
    top_k = 8  # Pull top 8 semantically similar matches

    # Step 1: Use Instructor to generate semantic embedding
    query_embedding = generate_semantic_query(
        prompt=input.prompt,
        theme=input.theme,
        main=input.main,
        sub=input.sub
    )

    # Step 2: Search each source and apply hybrid filter
    for source in input.sources:
        collection = client.get_or_create_collection(source, embedding_function=embedding_func)
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Hybrid Filtering: Require at least one keyword from user prompt
        query_string = input.prompt.lower()
        keywords = set(query_string.split())

        filtered_docs = []
        filtered_ids = []
        filtered_metadatas = []

        for doc, doc_id, metadata in zip(
            query_result['documents'][0],
            query_result['ids'][0],
            query_result['metadatas'][0]
        ):
            if any(word in doc.lower() for word in keywords):
                filtered_docs.append(doc)
                filtered_ids.append(doc_id)
                filtered_metadatas.append(metadata)

        # ✅ Limit to 3 results max
        filtered_docs = filtered_docs[:3]
        filtered_ids = filtered_ids[:3]
        filtered_metadatas = filtered_metadatas[:3]

        # Fallback to top 3 semantic if hybrid filter found nothing
        if not filtered_docs:
            print(f"⚠️ No hybrid matches for source: {source} — falling back to top 3 semantic.")
            filtered_docs = query_result['documents'][0][:3]
            filtered_ids = query_result['ids'][0][:3]
            filtered_metadatas = query_result['metadatas'][0][:3]

        # Final assignment
        query_result['documents'][0] = filtered_docs
        query_result['ids'][0] = filtered_ids
        query_result['metadatas'][0] = filtered_metadatas

        results.append({source: query_result})

    return results
