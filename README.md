# Local RAG Chatbot — Python

A local Python chatbot that answers questions from documents stored on your
Windows laptop.
## Hello
## Architecture

Windows documents -> text extraction -> chunks -> Ollama embeddings ->
local vector JSON -> similarity search -> Ollama chat -> answer + sources.

The application also searches arXiv using its public API, downloads PDF
papers, and automatically indexes them.

## Requirements

- Windows
- Python 3.11+
- Ollama

Install Ollama models:

```powershell
ollama pull gemma3:4b
ollama pull embeddinggemma
```

Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run:

```powershell
uvicorn app:app --reload
```

Open:

http://127.0.0.1:8000

## Add your documents

Put files here:

```text
documents/
    report.pdf
    policy.docx
    notes.txt
    handbook.md
```

Click **Ingest local documents**.

Supported formats:

- PDF
- DOCX
- TXT
- Markdown

Embeddings are saved locally in:

```text
data/vectors.json
```

## Download free public documents

Enter an arXiv search term such as:

```text
machine learning
```

and click **Download & index**.

The application will search arXiv, download matching PDF papers, extract their
text, and add them to the local knowledge base.

Respect arXiv API usage guidance and the license/terms applicable to each paper.

## Point to an existing Windows folder

For the first version, the default is the project's `documents` folder.
To use another location, change this line near the top of `app.py`:

```python
DOCUMENTS_DIR = BASE_DIR / "documents"
```

For example:

```python
DOCUMENTS_DIR = Path(r"C:\Users\YourName\Documents\MyKnowledgeBase")
```

## REST APIs

Ask a question:

```http
POST /api/chat
Content-Type: application/json

{"question":"What is the main idea?"}
```

Ingest local files:

```http
POST /api/ingest
```

Download from arXiv:

```http
POST /api/arxiv/download?query=machine%20learning&max_results=3
```

## Important limitations

This starter uses a JSON file and cosine similarity instead of a dedicated
vector database. It is suitable for a personal/small document collection.

Scanned image-only PDFs need OCR. The PDF extractor works when the PDF contains
selectable/extractable text.

Keep the server local unless you add authentication and appropriate security.

## Future upgrades

- OCR for scanned PDFs
- Automatic Windows folder watcher
- PostgreSQL + pgvector or Qdrant
- Conversation history
- Authentication
- More public-document APIs
- Document management UI
- Streaming responses
