"""Lightweight chunking and lexical tokenization for Chinese travel text."""

from __future__ import annotations

import re


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Prefer paragraph boundaries and use overlapping windows for long text."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""

    def append_window(value: str) -> None:
        start = 0
        step = chunk_size - overlap
        while start < len(value):
            piece = value[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_size >= len(value):
                break
            start += step

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            append_window(paragraph)
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = f"{prefix}\n\n{paragraph}".strip()
            if len(current) > chunk_size:
                append_window(current)
                current = ""

    if current:
        chunks.append(current)
    return chunks


def tokenize_search_text(text: str) -> list[str]:
    """Tokenize with CJK bigrams and lower-cased Latin terms."""

    tokens: list[str] = []
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    tokens.extend(token.lower() for token in _LATIN_TOKEN_RE.findall(text))
    return tokens


def build_search_text(text: str) -> str:
    """Return unique, whitespace-separated lexical terms."""

    return " ".join(dict.fromkeys(tokenize_search_text(text)))
