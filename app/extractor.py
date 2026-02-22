"""Document text extraction using Kreuzberg — with rich chunk metadata."""
import re
from kreuzberg import ExtractionConfig, OcrConfig, extract_bytes_sync


def extract_text(file_bytes: bytes, mime_type: str, enable_ocr: bool = True) -> dict:
    """Extract text and metadata from document bytes."""
    config = ExtractionConfig(
        enable_quality_processing=True,
        use_cache=True,
        ocr=OcrConfig(backend="tesseract", language="eng") if enable_ocr else None,
    )

    try:
        result = extract_bytes_sync(file_bytes, mime_type=mime_type, config=config)
        return {
            "content": result.content,
            "mime_type": result.mime_type,
            "metadata": dict(result.metadata) if result.metadata else {},
            "tables": [
                {"markdown": t.markdown, "page_number": t.page_number}
                for t in (result.tables or [])
            ],
            "char_count": len(result.content),
            "word_count": len(result.content.split()),
        }
    except Exception as e:
        return {
            "content": "",
            "mime_type": mime_type,
            "metadata": {},
            "tables": [],
            "char_count": 0,
            "word_count": 0,
            "error": str(e),
        }


# ── Heading / section detection ───────────────────────────────────────────────

# Patterns that likely indicate a section heading
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+.+|"                         # Markdown headings
    r"(?:[A-Z][A-Z\s\d\-]{2,60})\n|"        # ALL-CAPS lines
    r"(?:\d+\.)+\s+[A-Z][^\n]{3,80}\n|"     # Numbered headings  e.g. "1.2 Introduction"
    r"(?:Chapter|Section|Part)\s+[\dIVX]+[^\n]*\n"
    r")",
    re.MULTILINE,
)


def _detect_sections(text: str) -> list[dict]:
    """
    Return list of {title, start_char} marking where each section begins.
    Always inserts a sentinel at position 0 with title "".
    """
    sections = [{"title": "", "start_char": 0}]
    for m in _HEADING_RE.finditer(text):
        title = m.group(0).strip().lstrip("#").strip()
        if len(title) > 3:
            sections.append({"title": title[:120], "start_char": m.start()})
    return sections


def _section_at(sections: list[dict], char_pos: int) -> str:
    """Return the section title that covers char_pos."""
    title = ""
    for sec in sections:
        if sec["start_char"] <= char_pos:
            title = sec["title"]
        else:
            break
    return title


def _estimate_page(char_pos: int, total_chars: int, total_pages: int) -> int:
    """Linear interpolation page estimate (1-indexed)."""
    if total_chars == 0 or total_pages <= 1:
        return 1
    return max(1, round(char_pos / total_chars * total_pages) + 1)


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
    total_pages: int = 1,
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns list of dicts:
    {
        "text":          str,
        "chunk_index":   int,
        "start_char":    int,   # inclusive
        "end_char":      int,   # exclusive
        "page_estimate": int,   # 1-indexed
        "section":       str,   # nearest heading (may be "")
    }
    """
    if not text.strip():
        return []

    total_chars = len(text)
    sections = _detect_sections(text)

    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < total_chars:
        end = start + max_chars

        # Try to break at a natural boundary
        if end < total_chars:
            for sep in ["\n\n", ". ", "\n", " "]:
                pos = text.rfind(sep, start, end)
                if pos > start + overlap:
                    end = pos + len(sep)
                    break

        end = min(end, total_chars)
        chunk_text_val = text[start:end].strip()

        if chunk_text_val:
            chunks.append({
                "text": chunk_text_val,
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
                "page_estimate": _estimate_page(start, total_chars, total_pages),
                "section": _section_at(sections, start),
            })
            idx += 1

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1          # guard against infinite loop
        start = next_start

        if start >= total_chars:
            break

    return chunks


# ── MIME helpers ──────────────────────────────────────────────────────────────

MIME_TYPE_MAP = {
    "pdf":  "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc":  "application/msword",
    "txt":  "text/plain",
    "md":   "text/markdown",
    "html": "text/html",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
}


def guess_mime_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return MIME_TYPE_MAP.get(ext, "application/octet-stream")