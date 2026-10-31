"""ตรวจคำตอบที่ Claude Code เขียนไว้ใน out/answers_<split>.jsonl กับเฉลย (ไม่เรียก LLM)

แต่ละบรรทัด: {"qid": "Q01", "answer": "...", "citations": ["HR-01#2", ...]}
python -m rag.check_answers out/answers_dev.jsonl
"""
import json
import sys

from rag.evaluate import load_questions


def keywords_ok(answer: str, spec: str) -> bool:
    return all(any(alt.strip() in answer for alt in group.split("|")) for group in spec.split(";") if group)


def grade(q: dict, a: dict) -> dict:
    ans = a.get("answer", "")
    cited = {c.split("#")[0] for c in a.get("citations", [])}
    gold = set(filter(None, q["gold_doc_ids"].split(";")))
    r = {"qid": q["qid"], "type": q["type"], "keywords": keywords_ok(ans, q["answer_keywords"])}
    r["safe"] = not (q["must_not"] and q["must_not"] in ans)
    if q["type"] == "unanswerable":
        r["citation"] = True
    else:
        r["citation"] = bool(cited & gold)
    r["pass"] = r["keywords"] and r["safe"] and r["citation"]
    return r


def main(path: str):
    qs = {q["qid"]: q for q in load_questions("all")}
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    results = [grade(qs[a["qid"]], a) for a in rows]
    n = len(results)
    print(f"answers: {n}")
    for key in ["keywords", "citation", "safe", "pass"]:
        print(f"{key:9s}: {sum(r[key] for r in results)}/{n}")
    print("\nไม่ผ่าน:")
    for r in results:
        if not r["pass"]:
            print(f"  {r['qid']} ({r['type']}) keywords={r['keywords']} citation={r['citation']} safe={r['safe']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/answers_dev.jsonl")
