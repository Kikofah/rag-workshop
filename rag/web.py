"""เว็บแอปในเครื่องสำหรับค้นคืนระเบียบบริษัท ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.web [--port 8000]          เปิดเว็บ
python -m rag.web pending                 รายการคำถามจากเว็บที่ยังไม่มีคำตอบ
python -m rag.web answer "<คำถาม>" < ไฟล์คำตอบ   บันทึกคำตอบ (ใช้โดย skill /ask-web)

- GET /                     หน้าเว็บ (rag/static/index.html)
- GET /api/retrieve?q=..&k=4&method=hybrid   chunk ที่ค้นได้ (รวมฉบับ superseded ไว้เทียบ)
- GET /api/metrics          hit@1 / hit@4 / MRR ของชุด dev ทั้งสองวิธี
- GET /api/answer?q=..      คำตอบที่ Claude Code บันทึกไว้ (out/answers_web.jsonl หรือ out/answers_dev.jsonl)
- POST /api/questions       ส่งคำถามที่ยังไม่มีคำตอบเข้าคิว out/questions_web.jsonl
- POST /api/feedback        บันทึกความเห็นต่อคำตอบลง out/feedback.jsonl

เว็บไม่เรียก LLM: คำถามใหม่เข้าคิว แล้ว Claude Code ตอบด้วย /ask-web ตามขั้นตอนของ /ask
คำตอบเก็บใน out/answers_web.jsonl บรรทัดละ {"question": "...", "answer": "...", "citations": [...]}
หน้าเว็บถาม /api/answer ซ้ำจนคำตอบมา
"""
import argparse
import json
import re
import sys
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
QUEUE = OUT / "questions_web.jsonl"
FEEDBACK = OUT / "feedback.jsonl"
VOTES = ("up", "down", "none")
MAX_K = 10
MAX_BODY = 4096
MAX_QUESTION = 300
NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"
CITE_RE = re.compile(r"\[([A-Z]{2}-[\w-]+#\d+)\]")
_write_lock = Lock()


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, row: dict) -> None:
    OUT.mkdir(exist_ok=True)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean(question: str) -> str:
    return " ".join(str(question).split())


def pending_questions() -> list[dict]:
    """คำถามในคิวที่ยังไม่มีคำตอบ เรียงตามลำดับที่ส่งเข้ามา ไม่ซ้ำกัน

    id คือลำดับบรรทัดในไฟล์คิว (ไฟล์เขียนต่อท้ายอย่างเดียว id จึงไม่เปลี่ยน)
    """
    seen, out = set(), []
    for i, row in enumerate(_jsonl(QUEUE), start=1):
        key = _norm(row.get("question", ""))
        if key and key not in seen and saved_answer(row["question"]) is None:
            seen.add(key)
            out.append({**row, "id": i})
    return out


def queued_question(qid: int) -> str:
    rows = _jsonl(QUEUE)
    if not 1 <= qid <= len(rows):
        raise ValueError(f"ไม่มีคำถามหมายเลข {qid} ในคิว")
    return rows[qid - 1]["question"]


def enqueue(question: str) -> dict:
    q = _clean(question)
    if not q or len(q) > MAX_QUESTION:
        raise ValueError(f"คำถามต้องยาว 1–{MAX_QUESTION} ตัวอักษร")
    if saved_answer(q):
        return {"question": q, "status": "answered"}
    if all(_norm(r["question"]) != _norm(q) for r in pending_questions()):
        _append(QUEUE, {"at": _now(), "question": q})
    return {"question": q, "status": "pending"}


@lru_cache(maxsize=1)
def _chunk_ids() -> frozenset:
    return frozenset(c["chunk_id"] for c in get_retriever("tfidf").chunks)


def save_answer(question: str, answer: str) -> dict:
    """ตรวจรูปแบบแล้วบันทึกคำตอบที่ Claude Code เขียน (ทุก chunk_id ต้องมีจริง)"""
    q, a = _clean(question), answer.strip()
    if not q or not a:
        raise ValueError("ต้องมีทั้งคำถามและคำตอบ")
    cites = list(dict.fromkeys(CITE_RE.findall(a)))
    unknown = [c for c in cites if c not in _chunk_ids()]
    if unknown:
        raise ValueError(f"ไม่พบ chunk_id ในเอกสาร: {', '.join(unknown)}")
    if not cites and not a.startswith(NOT_FOUND):
        raise ValueError(f"คำตอบต้องอ้าง [chunk_id] อย่างน้อยหนึ่งรายการ หรือขึ้นต้นด้วย \"{NOT_FOUND}\"")
    row = {"question": q, "answer": a, "citations": cites, "at": _now()}
    _append(WEB_ANSWERS, row)
    return row


def save_feedback(data: dict) -> dict:
    vote = data.get("vote")
    question = str(data.get("question", "")).strip()
    if vote not in VOTES or not question:
        raise ValueError(f"ต้องมี question และ vote เป็น {', '.join(VOTES)}")
    row = {"at": _now(), "question": question[:500], "vote": vote, "method": str(data.get("method", ""))[:20]}
    _append(FEEDBACK, row)
    return row


POST_ROUTES = {
    "/api/feedback": save_feedback,
    "/api/questions": lambda d: enqueue(d.get("question", "")),
}


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
        elif url.path == "/api/pending":
            self._json({"pending": pending_questions()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        route = POST_ROUTES.get(urlparse(self.path).path)
        if route is None:
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
            self._json(route(data))
        except ValueError as e:  # รวม JSONDecodeError
            self._json({"error": str(e)}, 400)

    def log_message(self, fmt, *args) -> None:
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="เว็บแอปค้นคืนระเบียบบริษัท (ในเครื่อง)")
    ap.add_argument("--port", type=int, default=8000)
    # คำสั่งของ /ask-web อ้างคำถามด้วยหมายเลขคิว ข้อความจากเว็บจึงไม่ต้องผ่าน shell
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("pending", help="พิมพ์ <หมายเลข>\\t<คำถาม> ของคำถามจากเว็บที่ยังไม่มีคำตอบ")
    sub.add_parser("search", help="ค้นคืนด้วยคำถามในคิว (k=4 รวมฉบับเดิม) พิมพ์ JSON").add_argument("id", type=int)
    sub.add_parser("answer", help="บันทึกคำตอบของคำถามในคิวจาก stdin").add_argument("id", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "pending":
        for row in pending_questions():
            print(f"{row['id']}\t{row['question']}")
        return
    if a.cmd in ("search", "answer"):
        try:
            question = queued_question(a.id)
            if a.cmd == "search":
                res = retrieve(question, 4, "hybrid")
                print(json.dumps({"question": question, "results": res}, ensure_ascii=False, indent=2))
                return
            row = save_answer(question, sys.stdin.read())
        except ValueError as e:
            sys.exit(f"ทำไม่ได้: {e}")
        print(f"บันทึกแล้ว #{a.id}: {row['question']} · อ้าง {', '.join(row['citations']) or '-'}")
        return
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
