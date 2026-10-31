"""เว็บแอปในเครื่องสำหรับค้นคืนระเบียบบริษัท ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.web [--port 8000]

- GET /                     หน้าเว็บ (rag/static/index.html)
- GET /api/retrieve?q=..&k=4&method=hybrid   chunk ที่ค้นได้ (รวมฉบับ superseded ไว้เทียบ)
- GET /api/metrics          hit@1 / hit@4 / MRR ของชุด dev ทั้งสองวิธี
- GET /api/answer?q=..      คำตอบที่ /ask-eval บันทึกไว้ใน out/answers_dev.jsonl (ถ้าตรงกับคำถาม dev)

การสร้างคำตอบใหม่ยังทำโดย Claude Code ผ่าน /ask หน้าเว็บจึงมีปุ่มคัดลอกคำสั่งให้
"""
import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag.evaluate import ROOT, evaluate, load_questions
from rag.retrieve import METHODS, Retriever

STATIC = Path(__file__).resolve().parent / "static"
ANSWERS = ROOT / "out" / "answers_dev.jsonl"
MAX_K = 10


@lru_cache(maxsize=None)
def get_retriever(method: str) -> Retriever:
    return Retriever(method=method)


def retrieve(query: str, k: int = 4, method: str = "hybrid") -> list[dict]:
    k = max(1, min(k, MAX_K))
    res = get_retriever(method).search(query, k=k, include_superseded=True)
    return [{key: v for key, v in r.items() if key != "path"} for r in res]


@lru_cache(maxsize=1)
def dev_metrics() -> dict:
    """วัดเฉพาะชุด dev (กติกาข้อ 3) ใช้ค่าเริ่มต้นแบบเดียวกับ rag.evaluate"""
    qs = load_questions("dev")
    return {m: evaluate(get_retriever(m), qs, 4, include_superseded=m == "tfidf") for m in METHODS}


def saved_answer(query: str) -> dict | None:
    if not ANSWERS.exists():
        return None
    qid = next((q["qid"] for q in load_questions("dev") if q["question"].strip() == query.strip()), None)
    if qid is None:
        return None
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            if a.get("qid") == qid:
                return a
    return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        url = urlparse(self.path)
        p = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/retrieve":
            q, method = p.get("q", "").strip(), p.get("method", "hybrid")
            if not q:
                return self._json({"error": "ต้องระบุ q"}, 400)
            if method not in METHODS:
                return self._json({"error": f"method ต้องเป็น {', '.join(METHODS)}"}, 400)
            try:
                k = int(p.get("k", 4))
            except ValueError:
                return self._json({"error": "k ต้องเป็นตัวเลข"}, 400)
            self._json({"query": q, "method": method, "results": retrieve(q, k, method)})
        elif url.path == "/api/metrics":
            self._json(dev_metrics())
        elif url.path == "/api/answer":
            self._json({"answer": saved_answer(p.get("q", ""))})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args) -> None:
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="เว็บแอปค้นคืนระเบียบบริษัท (ในเครื่อง)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    for m in METHODS:  # สร้างดัชนีก่อนรับคำขอแรก
        get_retriever(m)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"เปิด http://127.0.0.1:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
