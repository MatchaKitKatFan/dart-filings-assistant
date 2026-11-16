import argparse
import json
import pathlib

DATA_DIR = pathlib.Path("data/dart")
CELL_TAGS = ["td", "th", "te", "tu"]


def table_to_text(table) -> str:
    """표를 '셀 | 셀 | 셀' 줄들로 바꾼다"""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(CELL_TAGS)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_report(xml: str) -> list[tuple[str, str]]:
    """사업보고서 XML → [(섹션 제목, 본문)]. TITLE 태그를 만날 때마다 새 섹션을 시작한다."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml, "html.parser")             # 태그 이름이 소문자로 바뀜
    sections, title, buf = [], "표지", []
    for el in soup.find_all(["title", "p", "table"]):
        if el.name == "title":
            if buf:
                sections.append((title, "\n".join(buf)))
            title, buf = el.get_text(" ", strip=True), []
        elif el.name == "table":
            buf.append(table_to_text(el))
        elif el.find_parent("table") is None:            # 표 안의 문단은 표에서 이미 처리
            text = el.get_text(" ", strip=True)
            if text:
                buf.append(text)
    if buf:
        sections.append((title, "\n".join(buf)))
    return sections


def split_report(sections: list[tuple[str, str]], source: str,
                 max_chars: int = 1200, overlap: int = 200) -> list[dict]:
    chunks = []
    for title, body in sections:
        for start in range(0, len(body), max_chars - overlap):
            piece = body[start:start + max_chars].strip()
            if len(piece) > 50:                      # 너무 짧은 조각은 버림
                chunks.append({"text": piece, "source": source, "title": title})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--collection", default="docs_1200")
    args = ap.parse_args()

    # 무거운 라이브러리는 여기서 import (테스트가 가볍게 돌도록)
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks = []
    for m in json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8")):
        xml = (DATA_DIR / m["file"]).read_text(encoding="utf-8")
        source = f"{m['corp']}_{m['year']}"                 # 예) 삼성전자_2024
        for c in split_report(parse_report(xml), source, args.chunk, args.overlap):
            chunks.append({**c, "corp": m["corp"], "year": m["year"]})
    print(f"청크 {len(chunks)}개")

    embedder = SentenceTransformer("BAAI/bge-m3")   # 처음 실행 시 약 2GB 다운로드
    embedder.max_seq_length = 1024
    vecs = embedder.encode([c["text"] for c in chunks], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=True)

    db = chromadb.PersistentClient(path="chroma")
    try:
        db.delete_collection(args.collection)          # 같은 이름이 있으면 새로 만듦
    except Exception:
        pass
    col = db.create_collection(args.collection, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 1000):
        part = chunks[i:i + 1000]
        col.add(
            ids=[f"c{j}" for j in range(i, i + len(part))],
            documents=[c["text"] for c in part],
            embeddings=vecs[i:i + 1000].tolist(),
            metadatas=[{"source": c["source"], "title": c["title"],
                        "corp": c["corp"], "year": c["year"]} for c in part],
        )
    print(f"'{args.collection}' 저장 완료")


if __name__ == "__main__":
    main()
