import datetime
import json
import pathlib
import uuid

LOG = pathlib.Path("logs/requests.jsonl")


def log_request(**fields):
    LOG.parent.mkdir(exist_ok=True)
    rec = {"id": uuid.uuid4().hex[:8],
           "ts": datetime.datetime.now().isoformat(timespec="seconds"), **fields}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
