import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent.loop import _get_retriever, run_agent
from llm import MODEL, client
from rag import answer


class State(TypedDict):
    question: str
    route: str
    answer: str
    history: Annotated[list, operator.add]   # 노드가 돌려준 리스트를 기존 값 뒤에 이어붙임


class Route(BaseModel):
    route: Literal["docs", "investigate"]
    reason: str


ROUTER = """질문을 분류하라.
docs: 사업보고서 한 곳만 찾아보면 답할 수 있는 사업 내용, 제품, 위험 요인 질문
investigate: 연도별 숫자 비교·계산, 최근 공시 확인처럼 여러 곳을 조사해야 하는 질문"""


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
    a = answer(state["question"], _get_retriever().filtered)["answer"]
    return {"answer": a, "history": _turn(state["question"], a)}


def agent_node(state: State) -> dict:
    res = run_agent(state["question"], history=state.get("history", []))
    return {"answer": res["answer"], "history": _turn(state["question"], res["answer"])}


g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("rag", rag_node)
g.add_node("agent", agent_node)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", lambda s: s["route"], {"docs": "rag", "investigate": "agent"})
g.add_edge("rag", END)
g.add_edge("agent", END)
graph_app = g.compile(checkpointer=MemorySaver())   # thread_id 별로 상태(대화 기록) 저장


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "local-test"}}
    while True:
        q = input("\n질문: ").strip()
        if not q:
            break
        out = graph_app.invoke({"question": q}, config=cfg)
        print(f"[{out['route']}] {out['answer']}")
