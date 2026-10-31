"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text

หลักการ
- หนึ่งหัวข้อ ## ต่อหนึ่ง chunk
- text ขึ้นต้นด้วย "<ชื่อเอกสาร> · <หัวข้อ>" เพื่อให้ chunk มีบริบทในตัวเองตอนค้นคืน
- ถ้าหัวข้อยาวเกิน max_chars ให้ตัดตามบรรทัด (บรรทัดเดียวที่ยาวเกินจะถูกตัดตามจำนวนตัวอักษร)
- metadata ทุกช่องของเอกสาร (ยกเว้น body) ถูกคัดลอกไปไว้ในทุก chunk
"""
import re

from rag.ingest import load_docs

MAX_CHARS = 600

_H2 = re.compile(r"^##\s+(.*)$")


def _sections(body: str) -> list[tuple[str, str]]:
    """แยก body เป็น [(ชื่อหัวข้อ, เนื้อหา)] ข้อความก่อน ## แรก (ไม่นับหัวเรื่อง #) ใช้ชื่อหัวข้อว่าง"""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        m = _H2.match(line)
        if m:
            sections.append((m.group(1).strip(), []))
        elif len(sections) == 1 and line.startswith("# "):
            continue  # หัวเรื่อง # ซ้ำกับ title ใน frontmatter อยู่แล้ว
        else:
            sections[-1][1].append(line)
    return [(title, "\n".join(lines).strip()) for title, lines in sections
            if "\n".join(lines).strip()]


def _split_lines(text: str, budget: int) -> list[str]:
    """รวมบรรทัดเป็นก้อนที่ยาวไม่เกิน budget ตัวอักษร"""
    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        while len(line) > budget:  # บรรทัดเดียวยาวเกิน ต้องตัดกลางบรรทัด
            pieces.append(line[:budget])
            line = line[budget:]
        pieces.append(line)

    parts: list[str] = []
    cur = ""
    for p in pieces:
        candidate = f"{cur}\n{p}" if cur else p
        if len(candidate) <= budget:
            cur = candidate
        else:
            parts.append(cur)
            cur = p
    if cur:
        parts.append(cur)
    return parts


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    if docs is None:
        docs = load_docs()

    chunks: list[dict] = []
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k != "body"}
        n = 0
        for section, content in _sections(doc["body"]):
            section = section or doc.get("title", "")
            header = f"{doc.get('title', '')} · {section}\n"
            for part in _split_lines(content, max_chars - len(header)):
                chunks.append({
                    **meta,
                    "chunk_id": f"{doc['doc_id']}#{n}",
                    "section": section,
                    "text": header + part,
                })
                n += 1
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
