import collections
import json
import sys

for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"evals/results/{name}.jsonl", encoding="utf-8")]
    t = collections.defaultdict(lambda: [0, 0, 0])      # 유형별 [문항 수, 정답 수, 모른다고 한 수]
    for r in rows:
        if r["type"] == "unanswerable":
            continue
        t[r["type"]][0] += 1
        t[r["type"]][1] += bool(r.get("correct"))
        t[r["type"]][2] += "찾을 수 없" in r["answer"]
    print(name, {k: f"정답 {c}/{n}, 모른다 {a}" for k, (n, c, a) in t.items()})
