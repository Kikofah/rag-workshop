"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag.chunk import chunk_docs

METHODS = ("tfidf",)


def _index_text(c: dict) -> str:
    """ข้อความที่ใช้ทำดัชนี: ชื่อเอกสาร + หัวข้อ + เนื้อหา"""
    return "\n".join([c.get("title", ""), c.get("section", ""), c["text"]])


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method not in METHODS:
            raise NotImplementedError(f"method={method!r} ยังไม่รองรับ (มีแค่ {', '.join(METHODS)})")
        self.method = method
        self.chunks = chunks if chunks is not None else chunk_docs()
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงใช้ n-gram ระดับตัวอักษรแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([_index_text(c) for c in self.chunks])

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        q = self.vectorizer.transform([query])
        # TfidfVectorizer ทำ L2 normalize แล้ว ผลคูณภายในจึงเท่ากับ cosine similarity
        scores = linear_kernel(q, self.matrix).ravel()
        results = []
        for i in np.argsort(-scores, kind="stable"):
            c = self.chunks[i]
            if not include_superseded and c.get("status") != "active":
                continue
            results.append({**c, "score": float(scores[i])})
            if len(results) >= k:
                break
        return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--include-superseded", action="store_true",
                    help="รวมเอกสารฉบับเก่า (status: superseded) ในผลค้น")
    ap.add_argument("--json", action="store_true", help="พิมพ์ผลเป็น JSON")
    a = ap.parse_args(argv)

    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for r in res:
        print(f"[{r['score']:.3f}] {r['chunk_id']} · {r['title']} · {r['section']} ({r['status']}, {r['effective_date']})")
        print("    " + r["text"].replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
