from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
import zipfile
import urllib.request
import shutil

app = FastAPI()

# CORS: Allow frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torah-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB paths
CHROMA_PATH = "/mnt/data/chromadb"
ZIP_PATH = "/mnt/data/chromadb.zip"
REMOTE_ZIP = "https://www.dropbox.com/scl/fi/xisf4ta1bik7o3jpkrj49/chromadb.zip?rlkey=syzwp7fpetsgh2bo9ropqzafw&st=0yvm3top&dl=1"

# Download and unzip if not already extracted
if not os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
    print("⬇️ Downloading chromadb.zip from Dropbox...")

    # Ensure /mnt/data exists
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)

    # Clean up any old ChromaDB if needed
    if os.path.exists(CHROMA_PATH):
        print("🧹 Removing old ChromaDB directory...")
        shutil.rmtree(CHROMA_PATH)

    urllib.request.urlretrieve(REMOTE_ZIP, ZIP_PATH)

    print("📦 Extracting chromadb.zip...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(CHROMA_PATH)

    print("✅ Unzip complete.")
else:
    print("📁 ChromaDB already exists on disk.")

# Initialize ChromaDB client
embedding_func = embedding_functions.DefaultEmbeddingFunction()
client = PersistentClient(path=CHROMA_PATH)

# Frontend source name → collection name map
COLLECTION_NAME_MAP = {
    "torah_texts": "torah_texts",
    "prophets_texts": "prophets_texts",
    "writings_texts": "writings_texts",
    "talmud_texts": "talmud_texts",
    "midrash_texts": "midrash_texts",
    "halacha_texts": "halacha_texts",
    "mitzvah_texts": "mitzvah_texts",
    "mussar_texts": "mussar_texts",
    "kabbalah_texts": "kabbalah_text",
    "chasidut_texts": "chassidut_text",
    "jewish_thought_texts": "jewish_thought_texts"
}

CITATION_TITLE_MAP = {
    "shaarei_teshuvah": "Shaarei Teshuvah",
    "tomer_devorah": "Tomer Devorah",
    "sefer_hayirah": "Sefer HaYirah",
    "orchot_tzadikim": "Orchot Tzadikim",
    "sefer_hayashar": "Sefer HaYashar",
    "iggeret_haramban": "Iggeret HaRamban",
    "iggeret_hagra": "Iggeret HaGra",
    "mesillat_yesharim": "Mesillat Yesharim",
    "sefer_hachinuch": "Sefer HaChinuch",
}

@app.get("/")
def root():
    return {"message": "Torah AI backend is live."}

class QueryRequest(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list

def format_citation(metadata):
    if metadata.get("book") == "Zohar" and metadata.get("parsha") and metadata.get("chapter"):
        return f"Zohar: {metadata['parsha']}, Chapter {metadata['chapter']}"
    elif metadata.get("start") and metadata.get("end"):
        return f"{metadata['start']}–{metadata['end']}"
    elif metadata.get("verse_range"):
        return metadata["verse_range"]
    elif metadata.get("book") and metadata.get("start"):
        return f"{metadata['book']} {metadata['start']}"
    elif metadata.get("citation"):
        raw = metadata["citation"]
        return CITATION_TITLE_MAP.get(raw.lower(), raw)
    elif metadata.get("book") and metadata.get("chapter"):
        return f"{metadata['book']}, Chapter {metadata['chapter']}"
    else:
        return "Unknown"

@app.post("/query")
async def query(request: QueryRequest):
    print(f"🔍 Received query: {request.prompt} | Sources: {request.sources}")
    responses = {}

    for source_name in request.sources:
        actual_collection = COLLECTION_NAME_MAP.get(source_name, source_name)

        try:
            collection = client.get_collection(name=actual_collection, embedding_function=embedding_func)
            print(f"✅ Using collection: {actual_collection} | Count: {collection.count()}")
        except Exception as e:
            print(f"⚠️ Skipping collection '{actual_collection}': {e}")
            continue

        results = collection.query(
            query_texts=[request.prompt],
            n_results=5,
            include=["metadatas", "documents"]
        )

        if not results["documents"] or not results["documents"][0]:
            print(f"⚠️ No documents found in '{actual_collection}' for this query.")
            continue

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        formatted = []
        for idx, (meta, doc) in enumerate(zip(metas, docs), start=1):
            text_en = meta.get("text_en", "").strip()
            text_he = meta.get("text_he", "").strip()

            if not text_en and not text_he:
                parts = doc.split("||")
                if len(parts) == 2:
                    text_en = parts[0].strip()
                    text_he = parts[1].strip()
                else:
                    text_en = doc.strip()
                    text_he = "(Hebrew unavailable)"
            elif not text_en:
                text_en = doc.strip().split("||")[0].strip()
            elif not text_he:
                parts = doc.split("||")
                if len(parts) > 1:
                    text_he = parts[1].strip()

            citation = format_citation(meta)

            formatted.append({
                "source_label": f"{actual_collection.replace('_texts', '').capitalize()} Source {idx}",
                "citation": citation,
                "text_en": text_en,
                "text_he": text_he
            })

        responses[f"{actual_collection}_responses"] = formatted

    return responses
