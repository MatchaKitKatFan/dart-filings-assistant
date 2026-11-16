import json
import random

from dart import financials

ACCOUNTS = ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계"]

meta = json.load(open("data/dart/meta.json", encoding="utf-8"))
have = {(m["corp"], m["year"]) for m in meta}
rows = []
for m in meta:
    try:
        accounts = financials(m["corp_code"], m["year"])
    except RuntimeError as e:                     # 013: 데이터 없음 등
        print(m["corp"], m["year"], e)
        continue
    fs = "CFS" if any(a["fs_div"] == "CFS" for a in accounts) else "OFS"   # 연결이 없으면 별도
    seen = set()
    for a in accounts:
        name = a["account_nm"].split("(")[0]          # "당기순이익(손실)" → "당기순이익"
        if a["fs_div"] != fs or name not in ACCOUNTS or name in seen:
            continue
        seen.add(name)
        try:
            won = int(a["thstrm_amount"].replace(",", ""))
        except ValueError:                         # 값이 "-" 등으로 비어 있음
            continue
        gold = [f"{m['corp']}_{m['year']}"]
        if (m["corp"], m["year"] + 1) in have:     # 다음 해 보고서에도 '전기' 값으로 나옴
            gold.append(f"{m['corp']}_{m['year'] + 1}")
        josa = "는" if name.endswith("계") else "은"   # 자산총계는 / 매출액은
        rows.append({"q": f"{m['corp']}의 {m['year']}년 {'연결' if fs == 'CFS' else '별도'} 기준 {name}{josa} 얼마야?",
                     "gold_sources": gold, "gold_answer": f"약 {won / 1e8:,.0f}억 원 ({won:,}원)",
                     "type": "number", "origin": "opendart-api"})

random.seed(0)
picked = random.sample(rows, min(20, len(rows)))
with open("evals/number_candidates.jsonl", "w", encoding="utf-8") as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"후보 {len(rows)}개 중 {len(picked)}개 저장")
