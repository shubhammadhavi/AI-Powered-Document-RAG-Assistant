"""Recursive character chunking: split on the biggest natural boundary that fits."""

from __future__ import annotations

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def _split_on(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # keep the separator attached to the left-hand piece so sentences stay readable
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split text into overlapping chunks of at most ``chunk_size`` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _atomic_pieces(text, chunk_size, separators)
    return _merge(pieces, chunk_size, chunk_overlap)


def _atomic_pieces(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Break text down until every piece fits inside chunk_size."""
    if len(text) <= chunk_size or not separators:
        return [text]

    separator, rest = separators[0], separators[1:]
    out: list[str] = []
    for part in _split_on(text, separator):
        if not part:
            continue
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(_atomic_pieces(part, chunk_size, rest))
    return out


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for piece in pieces:
        if length + len(piece) > chunk_size and current:
            chunks.append("".join(current).strip())
            # carry the tail of this chunk into the next one for context
            if chunk_overlap:
                tail: list[str] = []
                tail_len = 0
                for item in reversed(current):
                    if tail_len + len(item) > chunk_overlap:
                        break
                    tail.insert(0, item)
                    tail_len += len(item)
                current, length = tail, tail_len
            else:
                current, length = [], 0
        current.append(piece)
        length += len(piece)

    if current:
        tail_chunk = "".join(current).strip()
        if tail_chunk:
            chunks.append(tail_chunk)

    return [c for c in chunks if c]
