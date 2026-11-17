import json
import time

from agent.graph import classify
from agent.loop import run_agent
from llm import cost
from rag import answer
from retriever import Retriever

rows = [json.loads(l) for l in open("evals/scenarios.jsonl", encoding="utf-8") if l.strip()]
ret = Retriever("docs_1200")
stats = {"route_ok": 0, "tools_ok": 0, "steps": 0, "agent_cost": 0.0, "agent_sec": 0.0,
         "rag_cost": 0.0, "rag_sec": 0.0}

with open("evals/results/agent.jsonl", "w", encoding="utf-8") as f:
    for ex in rows:
        route = classify({"question": ex["q"]})["route"]
        stats["route_ok"] += route == ex["expect_route"]

        t = time.perf_counter()
        res = run_agent(ex["q"])
        stats["agent_sec"] += time.perf_counter() - t
        used = {tr["tool"] for tr in res["trace"]}
        stats["tools_ok"] += set(ex["expect_tools"]) <= used
        stats["steps"] += res["steps"]
        stats["agent_cost"] += res["usage"]["cost"]

        t = time.perf_counter()
        rag = answer(ex["q"], ret.rerank)                 # 같은 질문을 RAG로만 답하면?
        stats["rag_sec"] += time.perf_counter() - t
        if rag["usage"]:
            stats["rag_cost"] += cost(rag["usage"])

        f.write(json.dumps({"q": ex["q"], "route": route, "tools": sorted(used),
                            "agent_answer": res["answer"], "rag_answer": rag["answer"]},
                           ensure_ascii=False) + "\n")

n = len(rows)
print(f"라우팅 정확도 {stats['route_ok']/n:.2f} | 도구 선택 정확도 {stats['tools_ok']/n:.2f} | "
      f"평균 단계 {stats['steps']/n:.1f}")
print(f"에이전트: 평균 {stats['agent_sec']/n:.1f}s, ${stats['agent_cost']/n:.4f}/질문")
print(f"RAG만:   평균 {stats['rag_sec']/n:.1f}s, ${stats['rag_cost']/n:.4f}/질문")
