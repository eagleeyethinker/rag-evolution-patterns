import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
HANDBOOK_PATH = BASE_DIR.parent / "data" / "unstructured" / "text" / "company_handbook.txt"

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def load_handbook_sections(path):
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    documents = []
    for index, block in enumerate(blocks, start=1):
        documents.append(Document(page_content=block, metadata={"section": index, "source": path.name}))
    return documents


def stringify_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


documents = load_handbook_sections(HANDBOOK_PATH)
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
web_search = DuckDuckGoSearchRun()


@tool
def internal_policy_search(query: str) -> str:
    """Search internal company policies, budgets, and rules."""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n\n".join(
        f"Section {doc.metadata['section']}:\n{doc.page_content}"
        for doc in docs
    )


@tool
def web_search_tool(query: str) -> str:
    """Search the web for products, prices, and current external information."""
    return web_search.run(query)


agent = create_agent(
    model=llm,
    tools=[internal_policy_search, web_search_tool],
    system_prompt=(
        "You are a practical enterprise RAG agent. "
        "For internal budgets and policies, use internal_policy_search first. "
        "If the task also needs current market information, then use web_search_tool. "
        "Return a concise final answer with the internal budget and one or two matching products."
    ),
)

goal = "Find our internal budget limit for home office setups, then search the web to find a standing desk under that price."
result = agent.invoke({"messages": [{"role": "user", "content": goal}]})
final_message = result["messages"][-1]

print("Goal:", goal)
print("\nAnswer:\n", stringify_message_content(final_message.content))
