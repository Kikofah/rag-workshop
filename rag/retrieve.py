"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag.chunk import chunk_docs

METHODS = ("tfidf", "hybrid")

# น้ำหนักของ TF-IDF ตัวอักษรในคะแนนรวม (ที่เหลือเป็นของ BM25)
HYBRID_ALPHA = 0.7  # เลือกจากการ sweep บน dev: 0.6–0.8 ให้ผลเท่ากัน

# พจนานุกรมคำพ้องสำหรับขยายคำค้น สร้างจากคำถามชุด dev ที่ tfidf จัดอันดับพลาดเท่านั้น
# (ห้ามเพิ่มจากชุด test) คีย์คือคำที่ผู้ใช้พิมพ์ ค่าคือคำทางการที่ใช้ในเอกสาร
SYNONYMS = {
    "อีวี": "รถยนต์ไฟฟ้า",       # dev Q35
    "ev": "รถยนต์ไฟฟ้า",
    "เสียบชาร์จ": "จุดชาร์จ",    # dev Q35
    "พาสเวิร์ด": "รหัสผ่าน",     # dev Q24
    "password": "รหัสผ่าน",
    "หาย": "สูญหาย",             # dev Q30
}

_STOPWORDS = {"ที่", "ได้", "ไหม", "หรือ", "และ", "การ", "ของ", "แค่", "ไหน", "อะไร", "กี่", "เท่าไร", "ต้อง", "ให้", "จะ", "ไป"}


def _index_text(c: dict) -> str:
    """ข้อความที่ใช้ทำดัชนี: ชื่อเอกสาร + หัวข้อ + เนื้อหา"""
    return "\n".join([c.get("title", ""), c.get("section", ""), c["text"]])


def expand_query(query: str) -> str:
    """ต่อท้ายคำทางการจาก SYNONYMS เมื่อพบคำพ้องในคำค้น"""
    low = query.lower()
    extra = [v for k, v in SYNONYMS.items() if k in low and v not in query]
    return " ".join([query, *dict.fromkeys(extra)])


def tokenize(text: str) -> list[str]:
    """ตัดคำภาษาไทยด้วย pythainlp (newmm) ตัดช่องว่าง เครื่องหมาย และคำทั่วไปออก"""
    from pythainlp.tokenize import word_tokenize
    toks = word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)
    return [t for t in toks if t.strip() and any(ch.isalnum() for ch in t) and t not in _STOPWORDS]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


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
        if method == "hybrid":
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokenize(_index_text(c)) for c in self.chunks])

    def _scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            # TfidfVectorizer ทำ L2 normalize แล้ว ผลคูณภายในจึงเท่ากับ cosine similarity
            return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        query = expand_query(query)
        tf = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        bm = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        # สองวิธีมีสเกลต่างกัน จึง normalize เป็น 0–1 ก่อนรวม
        return HYBRID_ALPHA * _minmax(tf) + (1 - HYBRID_ALPHA) * _minmax(bm)

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        scores = self._scores(query)
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
    ap.add_argument("--method", default="tfidf", choices=METHODS)
    ap.add_argument("--include-superseded", action="store_true",
                    help="รวมเอกสารฉบับเก่า (status: superseded) ในผลค้น (ค่าเริ่มต้น: กรองออก)")
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
