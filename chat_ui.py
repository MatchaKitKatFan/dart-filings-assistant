import gradio as gr

from agent.graph import graph_app


def respond(message: str, history: list, request: gr.Request) -> str:
    # 브라우저 탭마다 다른 session_hash → 탭마다 별도의 대화방(thread_id)
    cfg = {"configurable": {"thread_id": request.session_hash}}
    out = graph_app.invoke({"question": message}, config=cfg)
    steps = " → ".join(f"`{t['tool']}({next(iter(t['input'].values()), '')})`"
                       for t in out.get("trace", []))
    return (f"{out['answer']}\n\n---\n"
            f"*경로: **{out['route']}** · 처리한 질문: {out['question']}*  \n"
            f"*도구: {steps or '없음'}*")


demo = gr.ChatInterface(
    respond,
    title="DART 기업 분석 에이전트",
    description="상장사 5곳(삼성전자, SK하이닉스, 현대자동차, NAVER, 카카오뱅크)의 사업보고서·재무·최근 공시를 조사해 답합니다.",
    examples=["SK하이닉스의 2025년 영업이익은 얼마야?",
              "그럼 2023년이랑 비교하면 얼마나 늘었어?",
              "NAVER가 최근에 낸 공시 중 IR 일정이 있어?"],
)

if __name__ == "__main__":
    demo.launch()
