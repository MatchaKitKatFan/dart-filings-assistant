import json
import pathlib
import re
import time

from dart import annual_reports, corp_codes, document

COMPANIES = ["삼성전자", "SK하이닉스", "현대자동차", "NAVER", "카카오뱅크"]
N_YEARS = 3                                                          # 회사마다 최근 3개 사업연도
OUT = pathlib.Path("data/dart")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes = corp_codes()
    meta = []
    for name in COMPANIES:
        if name not in codes:
            print(f"'{name}' 없음. 비슷한 이름:", [n for n in codes if name[:2] in n][:10])
            continue
        by_year = {}
        for rep in annual_reports(codes[name]):                      # 최근 접수순
            # A001 을 지정해도 분기·반기보고서가 섞여 오므로 이름으로 한 번 더 거른다
            m = re.search(r"사업보고서 \((\d{4})\.12\)", rep["report_nm"])   # "사업보고서 (2024.12)" → 2024
            if m:
                by_year.setdefault(int(m.group(1)), rep)             # 같은 해는 가장 최근 접수본
        for year in sorted(by_year, reverse=True)[:N_YEARS]:          # 사업연도 기준 최근 3개
            rep = by_year[year]
            path = OUT / f"{name}_{year}.xml"
            if not path.exists():                                    # 이미 받은 건 건너뜀
                path.write_text(document(rep["rcept_no"]), encoding="utf-8")
                time.sleep(1)                                        # 서버에 부담 주지 않기
            meta.append({"corp": name, "corp_code": codes[name], "year": year,
                         "rcept_no": rep["rcept_no"], "file": path.name})
            print(name, year, rep["rcept_no"], f"{path.stat().st_size // 1024:,}KB")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
