import re
from typing import List

def _sent_tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]

def split_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    sents = _sent_tokenize(text)
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for s in sents:
        if cur + len(s) + (1 if buf else 0) <= max_chars:
            buf.append(s)
            cur += len(s) + (1 if len(buf) > 1 else 0)
        else:
            if buf:
                chunk = " ".join(buf).strip()
                if chunk:
                    chunks.append(chunk)
                if overlap > 0:
                    keep = []
                    total = 0
                    for ss in reversed(buf):
                        if total + len(ss) + (1 if keep else 0) > overlap:
                            break
                        keep.append(ss)
                        total += len(ss) + (1 if keep else 0)
                    buf = list(reversed(keep))
                    cur = sum(len(x) for x in buf) + max(0, len(buf) - 1)
                else:
                    buf = []
                    cur = 0
            if len(s) >= max_chars:
                start = 0
                while start < len(s):
                    end = min(start + max_chars, len(s))
                    seg = s[start:end].strip()
                    if seg:
                        chunks.append(seg)
                    start = max(start + max_chars - overlap, end)
                buf = []
                cur = 0
            else:
                buf.append(s)
                cur = len(s)
    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
