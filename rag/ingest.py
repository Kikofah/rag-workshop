"""อ่านเอกสาร Markdown ที่มี frontmatter ใน data/docs"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"


def parse(text: str) -> tuple[dict, str]:
    """แยก frontmatter (บรรทัด key: value ระหว่าง ---) ออกจากเนื้อหา"""
    meta, body = {}, text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_docs(docs_dir: Path | str = DOCS_DIR) -> list[dict]:
    """คืน list ของเอกสาร: metadata ทุกช่อง + 'body' เรียงตาม doc_id"""
    docs = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        meta, body = parse(path.read_text(encoding="utf-8"))
        docs.append({**meta, "body": body, "path": str(path.relative_to(ROOT))})
    return docs
