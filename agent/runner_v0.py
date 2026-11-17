from anthropic import beta_tool

from agent.loop import SYSTEM, list_disclosures, search_docs, get_financials
from llm import MODEL, client

tools = [beta_tool(search_docs), beta_tool(get_financials), beta_tool(list_disclosures)]

question = input("질문: ")
runner = client.beta.messages.tool_runner(
    model=MODEL, max_tokens=8000, system=SYSTEM, tools=tools,
    messages=[{"role": "user", "content": question}],
)
last = None
for message in runner:
    for b in message.content:
        if b.type == "tool_use":
            print(f"[도구] {b.name} {b.input}")
    last = message
print("".join(b.text for b in last.content if b.type == "text"))
