from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
import json

app = FastAPI()

# CORS: Allow frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torah-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Writable path for ChromaDB on Render
CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chromadb")

# Updated: point to local data folder now inside backend
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Initialize ChromaDB client and embedding function
embedding_func = embedding_functions.DefaultEmbeddingFunction()
client = PersistentClient(path=CHROMA_PATH)

# Map frontend source names to ChromaDB collection names
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
    "chassidut_texts": "chassidut_text",
    "jewish_thought_texts": "jewish_thought_texts"
}

# Load documents only if collection is empty
def load_documents_if_empty():
    print(f"📁 Scanning for files in: {DATA_DIR}")
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                collection_name = os.path.splitext(file)[0].replace("_loadable", "").replace("_cleaned", "")
                try:
                    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)
                    if collection.count() == 0:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        documents, ids, metadatas = [], [], []

                        for i, entry in enumerate(data):
                            text = entry.get("text_en", "")
                            if not text:
                                continue
                            doc_id = f"{collection_name}_{i}"
                            documents.append(text)
                            ids.append(doc_id)
                            metadatas.append(entry)

                        if documents:
                            collection.add(documents=documents, ids=ids, metadatas=metadatas)
                            print(f"✅ Loaded {len(documents)} into '{collection_name}'")
                except Exception as e:
                    print(f"⚠️ Error loading {file}: {e}")

# Preload if needed
load_documents_if_empty()

@app.get("/")
def root():
    return {"message": "Torah AI backend is live."}

class QueryRequest(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list

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
