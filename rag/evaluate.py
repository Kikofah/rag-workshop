"""วัดผลการค้นคืนด้วยชุดคำถามที่มีเฉลย (ไม่เรียก LLM)

python -m rag.evaluate --split dev --method tfidf -k 4 [--min-hit 0.7]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

from rag.retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "eval" / "questions.csv"


def load_questions(split: str = "dev") -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split == "all" or r["split"] == split]


def evaluate(retriever, questions, k: int = 4, include_superseded: bool = True) -> dict:
    hits, hit1, rr, version_ok, version_n, per_type = 0, 0, 0.0, 0, 0, {}
    graded = [q for q in questions if q["gold_doc_ids"]]
    for q in graded:
        gold = set(q["gold_doc_ids"].split(";"))
        res = retriever.search(q["question"], k=10, include_superseded=include_superseded)
        ranks = [i for i, r in enumerate(res) if r["doc_id"] in gold]
        hit = bool(ranks) and ranks[0] < k
        hits += hit
        hit1 += bool(ranks) and ranks[0] == 0
        rr += 1 / (ranks[0] + 1) if ranks else 0
        t = per_type.setdefault(q["type"], [0, 0])
        t[0] += hit
        t[1] += 1
        if q["type"] == "version":
            version_n += 1
            version_ok += res[0]["status"] == "active"
    n = len(graded)
    return {
        "n": n,
        "hit@1": hit1 / n,
        f"hit@{k}": hits / n,
        "mrr": rr / n,
        "version_top1_active": version_ok / version_n if version_n else None,
        "per_type": {t: v[0] / v[1] for t, v in sorted(per_type.items())},
    }


def to_markdown(m: dict, k: int, label: str) -> str:
    lines = [f"### Retrieval eval · {label}", "", "| metric | value |", "|---|---|",
             f"| questions | {m['n']} |", f"| hit@1 | {m['hit@1']:.3f} |", f"| hit@{k} | {m[f'hit@{k}']:.3f} |", f"| MRR | {m['mrr']:.3f} |",
             f"| version questions, top-1 is active | {m['version_top1_active']:.3f} |", "",
             "| type | hit@k |", "|---|---|"]
    lines += [f"| {t} | {v:.3f} |" for t, v in m["per_type"].items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--min-hit", type=float, default=None)
    a = ap.parse_args()
    inc = a.include_superseded or a.method == "tfidf"
    m = evaluate(Retriever(method=a.method), load_questions(a.split), a.k, inc)
    md = to_markdown(m, a.k, f"{a.method} · {a.split}")
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if a.min_hit is not None and m[f"hit@{a.k}"] < a.min_hit:
        print(f"FAIL: hit@{a.k} {m[f'hit@{a.k}']:.3f} < {a.min_hit}")
        sys.exit(1)


if __name__ == "__main__":
    main()
