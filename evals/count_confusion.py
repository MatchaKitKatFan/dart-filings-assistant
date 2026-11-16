from evals.eval_retrieval import load
from retriever import Retriever

r = Retriever("docs_1200")
data = [ex for ex in load() if ex["gold_sources"]]
for method in ["vector", "filtered"]:
    n = 0
    for ex in data:
        top1 = getattr(r, method)(ex["q"], 1)[0]["source"]
        corp = top1.rsplit("_", 1)[0]
        gold_corps = {g.rsplit("_", 1)[0] for g in ex["gold_sources"]}
        if top1 not in ex["gold_sources"] and corp in gold_corps:   # 회사는 맞는데 연도가 틀림
            n += 1
    print(f"{method}: 1위가 같은 회사의 다른 해 보고서인 문항 {n}/{len(data)}")
