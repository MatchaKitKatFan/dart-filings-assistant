import gradio as gr

from fastapi import FastAPI
from pydantic import BaseModel

from agent.graph import chat as run_chat
from rag import answer
from chat_ui import demo
from retriever import Retriever, get_embedder

app = FastAPI(title="DART Filings Assistant")
retriever = Retriever("docs_1200")
get_embedder().encode(["워밍업"])      # 서버 시작 시 임베딩 모델을 미리 올려 첫 요청 지연(약 40초)을 없앰


class AskReq(BaseModel):
    question: str
    k: int = 5


class Source(BaseModel):
    n: int
    source: str
    title: str
    score: float


class AskRes(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=AskRes)
def ask_api(req: AskReq):
    r = answer(req.question, retriever.filtered, req.k)
    return {
        "answer": r["answer"],
        "sources": [{"n": n, "source": d["source"], "title": d["title"], "score": round(d["score"], 3)}
                    for n, d in enumerate(r["sources"], start=1)],
    }


class ChatReq(BaseModel):
    thread_id: str
    message: str


@app.post("/chat")
def chat(req: ChatReq):
    out = run_chat(req.thread_id, req.message)
    return {"answer": out["answer"], "route": out["route"],
            "question": out["question"], "trace": out.get("trace", [])}

app = gr.mount_gradio_app(app, demo, path="/ui")   # 채팅 화면을 같은 서버의 /ui 에 붙임
