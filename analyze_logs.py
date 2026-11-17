import collections
import json
import statistics

rows = [json.loads(l) for l in open("logs/requests.jsonl", encoding="utf-8")]
lat = [r["latency_ms"] for r in rows]

print(f"요청 {len(rows)}건 | 총비용 ${sum(r['cost'] for r in rows):.4f} | 질문당 ${statistics.mean(r['cost'] for r in rows):.4f}")
if len(lat) >= 2:
    q = statistics.quantiles(lat, n=20, method="inclusive")   # 실제 측정값 범위 안에서만 계산
    print(f"지연 p50 {statistics.median(lat):.0f}ms / p95 {q[18]:.0f}ms")
print("경로 비율:", dict(collections.Counter(r["route"] for r in rows)))
print("도구 사용 횟수:", dict(collections.Counter(t for r in rows for t in r["tools"])))
print("\n가장 느렸던 질문 3개")
for r in sorted(rows, key=lambda r: -r["latency_ms"])[:3]:
    print(f"  {r['latency_ms']}ms [{r['route']}] {r['question']}")
