from ingest import parse_report, split_report

XML = """<DOCUMENT><BODY>
<SECTION-1><TITLE>1. 사업의 개요</TITLE><P>당사는 반도체를 만든다.</P>
<TABLE><TR><TE>구분</TE><TE>매출액</TE></TR><TR><TD><P>DS</P></TD><TD>100</TD></TR></TABLE>
</SECTION-1><SECTION-1><TITLE>2. 주요 제품</TITLE><P>DRAM</P></SECTION-1>
</BODY></DOCUMENT>"""


def test_sections_split_by_title_and_table_not_duplicated():
    secs = parse_report(XML)
    assert [t for t, _ in secs] == ["1. 사업의 개요", "2. 주요 제품"]
    body = secs[0][1]
    assert "구분 | 매출액" in body and "DS | 100" in body
    assert body.count("DS") == 1               # 표 안의 <P>가 두 번 들어가지 않음


def test_chunks_respect_size_and_metadata():
    chunks = split_report([("T", "가" * 3000)], "a_2024", max_chars=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert all(c["source"] == "a_2024" and c["title"] == "T" for c in chunks)
