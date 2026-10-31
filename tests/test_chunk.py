from rag.chunk import chunk_docs
from rag.ingest import load_docs

REQUIRED = {"chunk_id", "doc_id", "title", "section", "status", "effective_date", "text"}


def test_every_section_has_a_chunk():
    docs = load_docs()
    chunks = chunk_docs(docs)
    n_sections = sum(d["body"].count("\n## ") + d["body"].startswith("## ") for d in docs)
    assert len(chunks) >= n_sections
    assert {c["doc_id"] for c in chunks} == {d["doc_id"] for d in docs}


def test_chunk_fields_and_size():
    chunks = chunk_docs()
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        assert REQUIRED <= set(c)
        assert c["text"].strip()
        assert len(c["text"]) <= 600
        assert c["status"] in {"active", "superseded"}


def test_section_title_is_kept():
    c = next(c for c in chunk_docs() if c["doc_id"] == "HR-03" and "ทันตกรรม" in c["section"])
    assert "3,000" in c["text"]
