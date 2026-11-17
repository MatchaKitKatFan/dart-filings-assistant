import operator
from typing import Annotated, Literal, TypedDict

import time
from request_log import log_request


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent.loop import _get_retriever, run_agent
from llm import MODEL, client, ask, cost
from rag import answer


class State(TypedDict):
    question: str
    route: str
    answer: str
    trace: list                             # 이번 질문에서 쓴 도구 기록 (매 질문마다 새로 덮어씀)
    cost: float                              # 이번 질문의 답변 비용 (달러)    
    history: Annotated[list, operator.add]   # 노드가 돌려준 리스트를 기존 값 뒤에 이어붙임


class Route(BaseModel):
    route: Literal["docs", "investigate"]
    reason: str


ROUTER = """질문을 분류하라.
docs: 사업보고서 한 곳만 찾아보면 답할 수 있는 사업 내용, 제품, 위험 요인 질문
investigate: 연도별 숫자 비교·계산, 최근 공시 확인처럼 여러 곳을 조사해야 하는 질문"""

REWRITE = """이전 대화를 참고해, 마지막 질문을 그 질문만 읽어도 뜻이 통하는 독립된 질문으로 다시 써라.
- <대화>는 오래된 것부터 최근 순서로 적혀 있다.
- "그 회사", "거기", "그럼", "작년" 같은 표현은 <직전 질문>에 나온 회사명과 연도로 바꾼다. 더 앞의 대화에 나온 회사로 바꾸지 않는다.
- 이미 독립된 질문이면 그대로 돌려준다.
- 다시 쓴 질문 한 문장만 출력한다."""


def rewrite(state: State) -> dict:
    history = state.get("history", [])
    if not history:                                  # 첫 질문은 바꿀 필요 없음 → LLM 호출 생략
        return {}
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])   # 최근 3턴, 턴당 500자
    last_q = history[-2]["content"]                  # 직전 턴의 질문 (기록은 [질문, 답] 순서로 쌓임)
    text, _ = ask(f"<대화>\n{convo}\n</대화>\n<직전 질문>{last_q}</직전 질문>\n\n"
                  f"마지막 질문: {state['question']}", system=REWRITE, max_tokens=300)
    return {"question": text.strip()}



def classify(state: State) -> dict:
    r = client.messages.parse(
        model=MODEL, max_tokens=1000, system=ROUTER,
        messages=[{"role": "user", "content": state["question"]}],
        output_format=Route,
    )
    return {"route": r.parsed_output.route}


def _turn(q: str, a: str) -> list:
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def rag_node(state: State) -> dict:
    r = answer(state["question"], _get_retriever().filtered)
    trace = [{"tool": "RAG 문서검색", "input": {"query": state["question"]}}]
    return {"answer": r["answer"], "trace": trace,
            "cost": cost(r["usage"]) if r["usage"] else 0.0,
            "history": _turn(state["question"], r["answer"])}


def agent_node(state: State) -> dict:
    res = run_agent(state["question"], history=state.get("history", []))
    used = ", ".join(f"{t['tool']}({next(iter(t['input'].values()), '')})" for t in res["trace"]) or "없음"
    note = f"\n\n(이 답변에서 사용한 도구: {used})"          # 다음 질문 때 에이전트가 근거를 알 수 있게
    return {"answer": res["answer"], "trace": res["trace"], "cost": res["usage"]["cost"],
            "history": _turn(state["question"], res["answer"] + note)}


g = StateGraph(State)
g.add_node("rewrite", rewrite)
g.add_node("classify", classify)
g.add_node("rag", rag_node)
g.add_node("agent", agent_node)
g.add_edge(START, "rewrite")
g.add_edge("rewrite", "classify")
g.add_conditional_edges("classify", lambda s: s["route"], {"docs": "rag", "investigate": "agent"})
g.add_edge("rag", END)
g.add_edge("agent", END)
graph_app = g.compile(checkpointer=MemorySaver())   # thread_id 별로 상태(대화 기록) 저장

def chat(thread_id: str, message: str) -> dict:
    """그래프를 한 번 실행하고 요청 로그를 남긴다. 채팅 화면과 /chat이 모두 이 함수를 쓴다."""
    if not message.strip():                      # 빈 질문: 그래프·LLM·로그 모두 건너뜀
        return {"answer": "질문을 입력해 주세요.", "route": "-", "question": "",
                "trace": [], "cost": 0.0}    
    t = time.perf_counter()
    out = graph_app.invoke({"question": message},
                           config={"configurable": {"thread_id": thread_id}})
    log_request(
        endpoint="chat", model=MODEL, thread_id=thread_id,
        question=message,                       # 4단계에서 mask_pii()로 감쌀 자리
        rewritten=out["question"], route=out["route"],
        tools=[tr["tool"] for tr in out.get("trace", [])],
        latency_ms=round((time.perf_counter() - t) * 1000),
        cost=round(out.get("cost", 0.0), 5),
    )
    return out



if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "local-test"}}
    while True:
        q = input("\n질문: ").strip()
        if not q:
            break
        out = graph_app.invoke({"question": q}, config=cfg)
        print(f"(처리한 질문: {out['question']})")
        print(f"[{out['route']}] {out['answer']}")
