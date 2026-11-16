import io
import os
import xml.etree.ElementTree as ET
import zipfile

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("DART_API_KEY")
BASE = "https://opendart.fss.or.kr/api"


def _get(path: str, **params) -> requests.Response:
    r = requests.get(f"{BASE}/{path}", params={"crtfc_key": KEY, **params}, timeout=120)
    r.raise_for_status()
    return r


def _json(path: str, **params) -> dict:
    data = _get(path, **params).json()
    if data.get("status") != "000":            # 000 = 정상, 013 = 데이터 없음
        raise RuntimeError(f"OpenDART 오류 {data.get('status')}: {data.get('message')}")
    return data


def _unzip(r: requests.Response) -> zipfile.ZipFile:
    if not r.content.startswith(b"PK"):        # zip 이 아니면 오류 메시지가 온 것
        raise RuntimeError(f"OpenDART 오류: {r.text[:200]}")
    return zipfile.ZipFile(io.BytesIO(r.content))


def corp_codes() -> dict[str, str]:
    """상장사 회사명 → 고유번호(8자리)"""
    with _unzip(_get("corpCode.xml")) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    return {it.findtext("corp_name").strip(): it.findtext("corp_code")
            for it in root.iter("list") if (it.findtext("stock_code") or "").strip()}


def annual_reports(corp_code: str, start: str = "20240101") -> list[dict]:
    """start 이후 접수된 사업보고서 목록 (정정된 경우 최종본만)"""
    return _json("list.json", corp_code=corp_code, bgn_de=start, pblntf_ty="A",
                 pblntf_detail_ty="A001", last_reprt_at="Y", page_count=100)["list"]


def document(rcept_no: str) -> str:
    """공시 원본 zip 에서 본문 XML 을 꺼내 문자열로 돌려준다"""
    with _unzip(_get("document.xml", rcept_no=rcept_no)) as z:
        names = z.namelist()
        raw = z.read(f"{rcept_no}.xml" if f"{rcept_no}.xml" in names else names[0])
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def financials(corp_code: str, year: int, reprt_code: str = "11011") -> list[dict]:
    """주요 재무 계정 (11011 = 사업보고서). 평가셋 정답과 2단계 에이전트 도구에 쓴다."""
    return _json("fnlttSinglAcnt.json", corp_code=corp_code, bsns_year=str(year),
                 reprt_code=reprt_code)["list"]


def disclosures(corp_code: str, start: str) -> list[dict]:
    """start(YYYYMMDD) 이후 접수된 모든 종류의 공시 목록 (최근 것부터 20건)"""
    return _json("list.json", corp_code=corp_code, bgn_de=start, page_count=20)["list"]
