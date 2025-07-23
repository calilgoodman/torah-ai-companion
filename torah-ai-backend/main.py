from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

app = FastAPI()

# ✅ CORS configured only for production frontend (Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torah-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client and embedding function
client = PersistentClient(path="chromadb")
embedding_func = embedding_functions.DefaultEmbeddingFunction()

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
    responses = {}

    for source_name in request.sources:
        try:
            collection = client.get_collection(name=source_name, embedding_function=embedding_func)
        except Exception as e:
            print(f"⚠️ Skipping unknown collection: {source_name} ({e})")
            continue

        results = collection.query(
            query_texts=[request.prompt],
            n_results=5,
            include=["metadatas", "documents"]
        )

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
                "source_label": f"{source_name.replace('_texts', '').capitalize()} Source {idx}",
                "citation": citation,
                "text_en": text_en,
                "text_he": text_he
            })

        responses[f"{source_name}_responses"] = formatted

    return responses
