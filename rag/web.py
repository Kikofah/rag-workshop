"""เว็บแอปในเครื่องสำหรับค้นคืนระเบียบบริษัท ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.web [--port 8000]

- GET /                     หน้าเว็บ (rag/static/index.html)
- GET /api/retrieve?q=..&k=4&method=hybrid   chunk ที่ค้นได้ (รวมฉบับ superseded ไว้เทียบ)
- GET /api/metrics          hit@1 / hit@4 / MRR ของชุด dev ทั้งสองวิธี
- GET /api/answer?q=..      คำตอบที่ Claude Code บันทึกไว้ (out/answers_web.jsonl หรือ out/answers_dev.jsonl)
- POST /api/feedback        บันทึกความเห็นต่อคำตอบลง out/feedback.jsonl

การสร้างคำตอบใหม่ยังทำโดย Claude Code ผ่าน /ask แล้วบันทึกลง out/answers_web.jsonl
บรรทัดละ {"question": "...", "answer": "...", "citations": [...]} หน้าเว็บมีปุ่มคัดลอกคำสั่งให้
"""
import argparse
import json
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

from rag.evaluate import ROOT, evaluate, load_questions
from rag.retrieve import METHODS, Retriever, expand_query

STATIC = Path(__file__).resolve().parent / "static"
OUT = ROOT / "out"
ANSWERS = OUT / "answers_dev.jsonl"
WEB_ANSWERS = OUT / "answers_web.jsonl"
FEEDBACK = OUT / "feedback.jsonl"
VOTES = ("up", "down", "none")
MAX_K = 10
MAX_BODY = 4096
_feedback_lock = Lock()


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


def _norm(text: str) -> str:
    return "".join(text.lower().split()).rstrip("?？")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def saved_answer(query: str) -> dict | None:
    """หาคำตอบที่บันทึกไว้ของคำถามนี้ (เทียบแบบไม่สนช่องว่าง) ไม่พบคืน None"""
    key = _norm(query)
    if not key:
        return None
    # บรรทัดหลังสุดชนะ เพื่อให้คำตอบที่ /ask เขียนทับทีหลังมีผล
    for a in reversed(_jsonl(WEB_ANSWERS)):
        if _norm(a.get("question", "")) == key:
            return {**a, "source": "web"}
    qid = next((q["qid"] for q in load_questions("dev") if _norm(q["question"]) == key), None)
    if qid is None:
        return None
    a = next((a for a in _jsonl(ANSWERS) if a.get("qid") == qid), None)
    return {**a, "question": query, "source": "dev"} if a else None


def save_feedback(data: dict) -> dict:
    vote = data.get("vote")
    question = str(data.get("question", "")).strip()
    if vote not in VOTES or not question:
        raise ValueError(f"ต้องมี question และ vote เป็น {', '.join(VOTES)}")
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": question[:500],
           "vote": vote, "method": str(data.get("method", ""))[:20]}
    OUT.mkdir(exist_ok=True)
    with _feedback_lock, FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


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
            expanded = expand_query(q) if method == "hybrid" else q
            self._json({"query": q, "expanded": expanded, "method": method, "results": retrieve(q, k, method)})
        elif url.path == "/api/metrics":
            self._json(dev_metrics())
        elif url.path == "/api/answer":
            self._json({"answer": saved_answer(p.get("q", ""))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/feedback":
            return self._json({"error": "not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", 0))
        except ValueError:
            size = -1
        if not 0 < size <= MAX_BODY:
            return self._json({"error": "ขนาดข้อมูลไม่ถูกต้อง"}, 400)
        try:
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("ต้องเป็น JSON object")
            self._json(save_feedback(data))
        except ValueError as e:  # รวม JSONDecodeError
            self._json({"error": str(e)}, 400)

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
