from rag.evaluate import evaluate, load_questions
from rag.retrieve import Retriever


def test_search_shape():
    res = Retriever(method="tfidf").search("ลาป่วยต้องมีใบรับรองแพทย์ไหม", k=4)
    assert len(res) == 4
    assert all("score" in r and "doc_id" in r for r in res)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_filter_superseded():
    res = Retriever(method="tfidf").search("ลาพักร้อนได้กี่วัน", k=6, include_superseded=False)
    assert all(r["status"] == "active" for r in res)


def test_baseline_dev_hit_at_4():
    m = evaluate(Retriever(method="tfidf"), load_questions("dev"), k=4)
    assert m["hit@4"] >= 0.9
