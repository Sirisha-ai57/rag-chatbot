from pathlib import Path
import json
import re
import math
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
VECTOR_FILE = DATA_DIR / "vectors.json"

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "gemma3:4b"
EMBED_MODEL = "embeddinggemma"

TOP_K = 12
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300

app = FastAPI(title="Document Chatbot")


class ChatRequest(BaseModel):
    question: str


def ensure_dirs():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def load_vectors():
    ensure_dirs()
    if not VECTOR_FILE.exists():
        return []
    return json.loads(VECTOR_FILE.read_text(encoding="utf-8"))


def save_vectors(vectors):
    ensure_dirs()
    VECTOR_FILE.write_text(
        json.dumps(vectors, ensure_ascii=False),
        encoding="utf-8"
    )


def extract_text(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith((".txt", ".md", ".text")):
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def chunk_text(text: str):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            paragraph = text.rfind("\n\n", start, end)
            sentence = text.rfind(". ", start, end)
            boundary = max(paragraph, sentence)

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def ollama_embed(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ollama_chat(system_prompt, user_prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": 4096
            }
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def cosine(a, b):
    if len(a) != len(b):
        return -1.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def ingest_file(path: Path):
    text = extract_text(path)

    if not text.strip():
        return 0

    pieces = chunk_text(text)
    embeddings = ollama_embed(pieces)

    vectors = load_vectors()
    source = str(path.resolve())

    vectors = [v for v in vectors if v["source"] != source]

    for index, (piece, embedding) in enumerate(zip(pieces, embeddings)):
        vectors.append({
            "id": f"{source}#{index}",
            "source": source,
            "text": piece,
            "embedding": embedding,
        })

    save_vectors(vectors)
    return len(pieces)


def ingest_local_documents():
    ensure_dirs()
    count = 0
    chunks = 0

    extensions = {".pdf", ".docx", ".txt", ".md", ".text"}

    for path in DOCUMENTS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            chunks += ingest_file(path)
            count += 1

    return {"files": count, "chunks": chunks}


def retrieve(question):
    vectors = load_vectors()

    if not vectors:
        return []

    query_embedding = ollama_embed([question])[0]

    scored = []
    for item in vectors:
        score = cosine(query_embedding, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:TOP_K]]


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/ingest")
def ingest():
    try:
        return ingest_local_documents()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not contact Ollama: {exc}"
        )


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    try:
        matches = retrieve(request.question)

        if not matches:
            return {
                "answer": (
                    "I don't have any indexed documents yet. "
                    "Put documents in the documents folder and click Ingest."
                ),
                "sources": [],
            }

        context = "\n\n---\n\n".join(
            f"SOURCE: {item['source']}\n{item['text']}"
            for item in matches
        )

        system = (
            "You are a document question-answering assistant.\n"
            "Answer ONLY from the supplied context.\n"
            "If the context does not contain the answer, say: "
            "\"I don't know based on the indexed documents.\"\n"
            "Do not invent facts or sources.\n"
            "Be concise but clear."
        )

        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{request.question}"

        answer = ollama_chat(system, prompt)

        sources = list(dict.fromkeys(item["source"] for item in matches))

        return {"answer": answer, "sources": sources}

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not contact Ollama: {exc}"
        )


@app.post("/api/arxiv/download")
def download_arxiv(query: str, max_results: int = 3):
    max_results = max(1, min(max_results, 10))

    encoded = urllib.parse.quote(query)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&start=0&max_results={max_results}"
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "LocalRagChatbot/1.0"},
            timeout=30,
        )
        response.raise_for_status()

        root_xml = ET.fromstring(response.text)
        namespace = {"a": "http://www.w3.org/2005/Atom"}

        downloaded = []

        for entry in root_xml.findall("a:entry", namespace):
            title = entry.findtext("a:title", default="", namespaces=namespace)
            title = re.sub(r"\s+", " ", title).strip()

            pdf_url = None

            for link in entry.findall("a:link", namespace):
                if link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                    break

            if not pdf_url:
                continue

            safe_title = re.sub(r"[^a-zA-Z0-9._-]+", "_", title)
            safe_title = safe_title[:90] or "arxiv-paper"
            output = DOCUMENTS_DIR / f"{safe_title}.pdf"

            pdf_response = requests.get(
                pdf_url,
                headers={"User-Agent": "LocalRagChatbot/1.0"},
                timeout=120,
            )
            pdf_response.raise_for_status()

            output.write_bytes(pdf_response.content)
            ingest_file(output)

            downloaded.append(str(output.resolve()))

        return {"count": len(downloaded), "downloaded": downloaded}

    except (requests.RequestException, ET.ParseError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
