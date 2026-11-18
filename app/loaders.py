"""Turn a file on disk into page-level text. Only hard dependency is pypdf (for PDFs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    text: str
    page: int  # 1-based; 1 for non-paged formats


class UnsupportedFile(Exception):
    pass


class EmptyDocument(Exception):
    pass


def _read_text_file(path: Path) -> str:
    """Read a text file without blowing up on odd encodings (the old TextLoader did)."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _read_pdf(path: Path) -> list[Page]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedFile(
            "Reading PDFs needs pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # many PDFs are 'encrypted' with an empty password
        except Exception as exc:
            raise UnsupportedFile(
                f"{path.name} is password protected, so its text cannot be read."
            ) from exc

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(Page(text=text, page=number))
    return pages


def load_file(path: str | Path) -> list[Page]:
    """Load a document into pages of text.

    Raises UnsupportedFile for formats we do not handle and EmptyDocument when a
    file carries no extractable text (a scanned PDF, for instance).
    """
    path = Path(path)
    if not path.is_file():
        raise UnsupportedFile(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _read_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = _read_text_file(path)
        pages = [Page(text=text, page=1)] if text.strip() else []
    else:
        raise UnsupportedFile(
            f"{suffix or 'that file type'} is not supported. Use PDF, TXT or MD."
        )

    if not pages:
        raise EmptyDocument(
            f"No text could be extracted from {path.name}. "
            "If it is a scanned PDF it needs OCR first."
        )
    return pages
