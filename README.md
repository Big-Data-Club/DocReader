# 📚 BDC Doc Reader

<div align="center">

**Hệ thống đọc và khai thác tài liệu thông minh — Dự án của Big Data Club · HCMUT**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## 📖 Giới thiệu

**BDC Doc Reader** là một hệ thống **Personal Knowledge Base** mã nguồn mở, được phát triển bởi [**Big Data Club (BDC)**](https://www.facebook.com/bigdataclubhcmut) — Câu lạc bộ Dữ liệu lớn trực thuộc Trường Đại học Bách Khoa TP.HCM (HCMUT).

Dự án được xây dựng với mục tiêu nghiên cứu và ứng dụng các kỹ thuật **Retrieval-Augmented Generation (RAG)** và **Knowledge Graph** vào bài toán khai thác tri thức từ tài liệu cá nhân — hỗ trợ đa ngôn ngữ Việt–Anh, không phụ thuộc vào dịch vụ cloud đắt tiền.

### ✨ Điểm nổi bật

- 🌏 **Đa ngôn ngữ Việt–Anh** — Detect ngôn ngữ, dịch query tự động, tìm kiếm song song, RRF fusion
- 🕸️ **GraphRAG** — Knowledge Graph với entity aliases, community detection, 2-hop traversal
- 🧠 **Bộ nhớ hội thoại** — Nén lịch sử, entity threading, personalization theo sở thích người dùng
- 🔍 **Pipeline RAG đa tầng** — Multi-query · HyDE · Step-back · Hypothetical questions (VI+EN)
- 📦 **Hoàn toàn self-hosted** — MinIO · ChromaDB · Kuzu chạy trên máy của bạn
- ⚡ **LLM miễn phí** — Groq API (free tier đủ dùng cho cá nhân)

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        UPLOAD PIPELINE                      │
│                                                             │
│  File (PDF/DOCX/IMG...)                                     │
│       │                                                     │
│       ├──► MinIO ──────────────── lưu file gốc (S3)         │
│       │                                                     │
│       ├──► Kreuzberg (OCR) ─────► text chunks               │
│       │         │                      │                    │
│       │         │              ┌───────▼────────┐           │
│       │         │              │   ChromaDB     │           │
│       │         │              │ (vector index) │           │
│       │         │              │  VI+EN queries │           │
│       │         │              └───────▲────────┘           │
│       │         │                      │ embed              │
│       │         └──► Groq LLM ─────────┤                    │
│       │                   │      bilingual questions        │
│       │                   │                                 │
│       └──────────────► Kuzu Graph                           │
│                          │  entities + relations            │
│                          │  aliases (VI↔EN)                 │
│                          │  COOCCURS_WITH edges             │
│                          └─► community detection            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                       │
│                                                             │
│  User Query (VI or EN)                                      │
│       │                                                     │
│       ├─ detect language ──────────────────────────────┐    │
│       │                                                │    │
│       ├─ translate + expand (multi-query · HyDE ·      │    │
│       │   step-back) in BOTH VI and EN                 │    │
│       │                                                │    │
│       ├─ ChromaDB search (VI queries) ──► ranked_vi    │    │
│       ├─ ChromaDB search (EN queries) ──► ranked_en    │    │
│       │               └────── RRF merge ──────────┐    │    │
│       │                                           │    │    │
│       ├─ Hypothetical question index (VI+EN) ─────┤    │    │
│       │                                           │    │    │
│       ├─ Kuzu graph traversal (2-hop, aliases) ───┤    │    │
│       │                                           │    │    │
│       └─ Global community fallback ───────────────┘    │    │
│                           │                            │    │
│                    RRF final merge                     │    │
│                           │                            │    │
│                  LLM Reranker (+ profile boost)        │    │
│                           │                            │    │
│                  Groq LLM answer ◄─ entity context     │    │
│                           │         history context    │    │
│                           │         user interests     │    │
│                           ▼                            │    │
│                  Answer in user's language ◄───────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Công nghệ | Vai trò |
|---|---|---|
| **Storage** | [MinIO](https://min.io) | Lưu file gốc — S3-compatible object storage |
| **Extraction** | [Kreuzberg](https://github.com/Goldziher/kreuzberg) | Parse PDF, DOCX, XLSX, PPTX, ảnh (OCR) |
| **Vector DB** | [ChromaDB](https://www.trychroma.com) | Semantic search — embedded, không cần server riêng |
| **Graph DB** | [Kuzu](https://kuzudb.com) | Knowledge graph — embedded, không cần server riêng |
| **Embeddings** | [sentence-transformers](https://sbert.net) `paraphrase-multilingual-MiniLM-L12-v2` | Local, free, hỗ trợ 50+ ngôn ngữ kể cả Việt |
| **LLM (mạnh)** | Groq `llama-3.3-70b-versatile` | RAG answer, reranking, entity summarization |
| **LLM (nhanh)** | Groq `llama-3.1-8b-instant` | KG extraction, query expansion, translation |
| **TTS** | Groq `canopylabs/orpheus-v1-english` | Text to speech |
| **API** | [FastAPI](https://fastapi.tiangolo.com) | REST backend |
| **History** | SQLite + ChromaDB | Lưu hội thoại, index ngữ nghĩa, user profiling |

---

## 📁 Cấu trúc dự án

```
bdc-doc-reader/
│
├── app/                        # Application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, tất cả API endpoints
│   ├── config.py               # Settings (pydantic-settings, .env)
│   │
│   ├── extractor.py            # Text extraction + chunking (Kreuzberg)
│   ├── storage.py              # MinIO document storage
│   ├── vectorstore.py          # ChromaDB — multilingual search, RRF
│   ├── graph.py                # Kuzu knowledge graph — entities, aliases,
│   │                           #   communities, 2-hop traversal
│   ├── history.py              # Conversation memory — SQLite + ChromaDB
│   │                           #   dual-index, compression, user profiling
│   ├── multilingual.py         # Language detection, translation,
│   │                           #   cross-lingual query expansion, RRF merge
│   └── groq_client.py          # Groq API — LLM, TTS, KG extraction,
│                               #   bilingual question generation
│
├── static/
│   └── index.html              # Single-page frontend (vanilla JS)
│
├── data/                       # Runtime data (gitignored)
│   ├── chroma/                 # ChromaDB vector index
│   ├── kuzu/                   # Kuzu graph database
│   └── history.db              # SQLite conversation history
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # Dependencies (uv/pip)
├── .env.example
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu

| Thứ cần có | Version |
|---|---|
| Python | ≥ 3.11 |
| Docker + Docker Compose | bất kỳ version gần đây |
| Groq API Key | Free tại [console.groq.com](https://console.groq.com/keys) |
| RAM | ≥ 4 GB (embedding model ~420 MB) |
| Disk | ≥ 2 GB free |

---

### Cách 1 — Docker Compose *(khuyến nghị)*

Toàn bộ stack (app + MinIO) chạy trong container, không cần cài gì thêm ngoài Docker.

```bash
# 1. Clone repo
git clone https://github.com/bigdataclub-hcmut/bdc-doc-reader.git
cd bdc-doc-reader

# 2. Tạo file .env
cp .env.example .env
```

Mở `.env`, điền Groq API key:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

```bash
# 3. Build và khởi động
docker compose up --build

# Lần đầu mất ~5-10 phút (download embedding model ~420 MB)
# Từ lần 2 trở đi khởi động trong vài giây
```

| Dịch vụ | URL |
|---|---|
| 🌐 App | http://localhost:8000 |
| 🗄️ MinIO Console | http://localhost:9001 (admin: `minioadmin` / `minioadmin`) |
| 📖 API Docs | http://localhost:8000/docs |

```bash
# Dừng
docker compose down

# Dừng + xóa toàn bộ data (khi đổi embedding model)
docker compose down -v
```

> ⚠️ **Đổi embedding model?** Phải chạy `docker compose down -v` rồi build lại — ChromaDB index không tương thích giữa các model khác nhau.

---

### Cách 2 — Local development với `uv`

Dùng khi bạn muốn chỉnh sửa code và thấy thay đổi ngay lập tức.

**Bước 1 — Cài uv** (package manager nhanh hơn pip)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Bước 2 — Cài system dependencies**

Ubuntu/Debian:
```bash
sudo apt update && sudo apt install -y \
    pandoc \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-vie \
    libgl1
```

macOS:
```bash
brew install pandoc tesseract tesseract-lang
```

Windows: Cài [Pandoc](https://pandoc.org/installing.html) và [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) thủ công.

**Bước 3 — Cài Python dependencies**
```bash
uv sync
```

**Bước 4 — Khởi động MinIO** (cần Docker)
```bash
docker run -d \
  --name bdc-minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

**Bước 5 — Cấu hình môi trường**
```bash
cp .env.example .env
```

Chỉnh `.env` cho local:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

DATA_DIR=./data
CHROMA_DIR=./data/chroma
KUZU_DIR=./data/kuzu
```

**Bước 6 — Tạo thư mục data**
```bash
mkdir -p data/chroma data/kuzu
```

**Bước 7 — Chạy app**
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`--reload` tự restart server khi bạn lưu file — rất tiện khi dev.

---

## 🎯 Hướng dẫn sử dụng

### Upload tài liệu
1. Mở http://localhost:8000
2. Click **Upload** → chọn file (PDF, DOCX, TXT, MD, XLSX, PPTX, PNG, JPG)
3. Hệ thống tự động:
   - Extract text (có OCR nếu là ảnh/scan)
   - Chunk và embed vào ChromaDB
   - Build Knowledge Graph (entities + relations + aliases VI/EN)
   - Index bilingual hypothetical questions *(background)*
   - Detect community và tóm tắt *(background)*

### RAG Query
- Gõ câu hỏi bằng **tiếng Việt hoặc tiếng Anh** — hệ thống tự detect và dịch
- Kết quả trả về trong ngôn ngữ của câu hỏi
- Citation `[1]`, `[2]`... link tới đúng đoạn văn nguồn

### Các tính năng nâng cao
| Tính năng | Mô tả |
|---|---|
| **Reasoning mode** | LLM phân tích step-by-step trước khi trả lời |
| **Speak Answer** | TTS đọc câu trả lời (10 giọng) |
| **Structured Extraction** | Định nghĩa JSON schema, extract dữ liệu có cấu trúc |
| **Knowledge Graph viewer** | Xem entities, relations, community của từng tài liệu |
| **Entity detail** | Click vào entity → xem summary, aliases, lịch sử hội thoại liên quan |

---

## 📡 API Reference

### Documents
```
POST   /api/upload                    Upload và index tài liệu mới
GET    /api/documents                 Danh sách tất cả tài liệu
DELETE /api/documents/{doc_id}        Xóa tài liệu
```

### Query
```
POST   /api/query                     RAG query (multilingual GraphRAG)
GET    /api/passage/{doc_id}/{idx}    Xem chunk gốc với context xung quanh
```

### Conversations
```
GET    /api/conversations             Danh sách hội thoại
POST   /api/conversations             Tạo hội thoại mới
GET    /api/conversations/{id}        Chi tiết + lịch sử tin nhắn
PATCH  /api/conversations/{id}        Đổi tên hội thoại
DELETE /api/conversations/{id}        Xóa hội thoại
```

### Knowledge Graph
```
GET    /api/graph/{doc_id}            Graph data của tài liệu
GET    /api/entity/{entity_id}        Chi tiết entity (neighborhood, aliases, history)
```

### User Profile
```
GET    /api/user/interests            Top topics người dùng quan tâm
GET    /api/user/profile              Full user profile (interests + language preference)
```

### Utilities
```
POST   /api/extract-structured        Structured JSON extraction
POST   /api/tts                       Text to speech
GET    /api/tts/voices                Danh sách giọng đọc
GET    /api/health                    Health check
GET    /docs                          Swagger UI (auto-generated)
```

**Ví dụ query request:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Các thuật toán học máy nào được đề cập trong tài liệu?",
    "use_reasoning": false,
    "n_results": 5,
    "use_graph_traversal": true
  }'
```

---

## ⚙️ Cấu hình

Tất cả cấu hình qua file `.env` (hoặc environment variables):

| Biến | Mặc định | Mô tả |
|---|---|---|
| `GROQ_API_KEY` | *(bắt buộc)* | API key từ console.groq.com |
| `GROQ_CHAT_MODEL` | `llama-3.3-70b-versatile` | Model cho RAG answer |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Model cho KG extraction, query rewrite |
| `GROQ_TTS_MODEL`  | `canopylabs/orpheus-v1-english` | Text-to-speech model |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (⚠️ đổi → phải re-index) |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `documents` | Tên bucket lưu tài liệu |
| `DATA_DIR` | `/app/data` | Thư mục data gốc |
| `CHROMA_DIR` | `/app/data/chroma` | ChromaDB path |
| `KUZU_DIR` | `/app/data/kuzu` | Kuzu graph DB path |

**Lựa chọn embedding model** (đổi trong `config.py`):

| Model | Size | Dim | Chất lượng |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` *(mặc định)* | ~420 MB | 384 | Tốt — nhanh |
| `paraphrase-multilingual-mpnet-base-v2` | ~970 MB | 768 | Tốt hơn — chậm hơn |
| `BAAI/bge-m3` | ~570 MB | 1024 | Tốt nhất — nặng hơn |

---

## 🤝 Contribution Guide

Chúng tôi rất hoan nghênh mọi đóng góp từ cộng đồng! Dưới đây là quy trình chuẩn.

### Quy trình đóng góp

```
fork repo → clone về máy → tạo branch mới → code → test → PR
```

**1. Fork và clone**
```bash
# Fork trên GitHub UI, rồi:
git clone https://github.com/<your-username>/bdc-doc-reader.git
cd bdc-doc-reader
git remote add upstream https://github.com/bigdataclub-hcmut/bdc-doc-reader.git
```

**2. Tạo branch mới** — đặt tên theo convention:
```bash
# Feature mới
git checkout -b feat/multilingual-ocr

# Fix bug
git checkout -b fix/graph-traversal-depth

# Cải thiện docs
git checkout -b docs/update-readme

# Refactor
git checkout -b refactor/vectorstore-rrf
```

**3. Setup môi trường dev**
```bash
uv sync
cp .env.example .env
# Điền GROQ_API_KEY vào .env
```

**4. Code và test thủ công**
```bash
# Chạy app với hot-reload
uv run uvicorn app.main:app --reload

# Kiểm tra code style
uv run ruff check app/
uv run ruff format app/
```

**5. Commit** — theo [Conventional Commits](https://www.conventionalcommits.org/):
```bash
git add .
git commit -m "feat(graph): add entity alias cross-lingual resolution"
git commit -m "fix(history): prevent duplicate compression on short convs"
git commit -m "docs: update API reference for v0.6 endpoints"
```

**6. Sync với upstream trước khi push**
```bash
git fetch upstream
git rebase upstream/main
git push origin feat/your-feature-name
```

**7. Mở Pull Request** trên GitHub — điền đầy đủ template PR:
- Mô tả ngắn gọn thay đổi là gì
- Lý do thay đổi (link issue nếu có)
- Cách test thủ công
- Screenshots nếu có thay đổi UI

---

### Những gì cần giúp đỡ

Xem [Issues](https://github.com/bigdataclub-hcmut/bdc-doc-reader/issues) để tìm task. Một số hướng đóng góp:

| Loại | Ví dụ |
|---|---|
| 🐛 **Bug fix** | Sửa lỗi ChromaDB khi index rỗng, fix encoding tiếng Việt trong OCR |
| ✨ **Feature** | Thêm hỗ trợ file mới, cải thiện UI, thêm export conversation |
| 🔬 **Research** | Thử nghiệm embedding model mới, cải thiện chunking strategy |
| 📖 **Docs** | Viết tutorial, cải thiện docstring, dịch README sang tiếng Anh |
| 🧪 **Testing** | Viết unit test cho extractor, vectorstore, graph |
| 🎨 **UI/UX** | Cải thiện frontend (vanilla JS trong `static/index.html`) |

---

### Code conventions

```
app/
├── Mỗi module có docstring đầu file giải thích mục đích
├── Type hints bắt buộc cho tất cả function public
├── Exception handling: không để lỗi crash server — log + fallback
├── Tên biến/function: snake_case tiếng Anh
└── Comment giải thích "why", không phải "what"
```

**Cấu trúc function:**
```python
def my_function(param: str, optional: int = 5) -> list[dict]:
    """
    Một câu mô tả ngắn gọn chức năng.

    Args / Returns chỉ cần ghi khi không self-explanatory.
    """
    ...
```

---

## 👥 Về Big Data Club · HCMUT

**Big Data Club (BDC)** là câu lạc bộ học thuật trực thuộc Trường Đại học Bách Khoa TP.HCM, tập hợp các bạn sinh viên yêu thích **Data Science, Machine Learning, và AI**.

- 🌐 Facebook: [facebook.com/BDCofHCMUT](https://www.facebook.com/BDCofHCMUT)
- 📧 Email: bdc@hcmut.edu.vn
- 🏫 Địa chỉ: 268 Lý Thường Kiệt, TP.HCM

Dự án này là một phần trong chương trình **BDC Research Projects** — nơi thành viên câu lạc bộ cùng nhau nghiên cứu và xây dựng các hệ thống AI thực tế phục vụ giáo dục.

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).  
Sử dụng tự do cho mục đích học thuật, nghiên cứu và phi thương mại.

---

<div align="center">

Made with ❤️ by **Big Data Club · HCMUT**

*"Learning by building — Building by sharing"*

</div>