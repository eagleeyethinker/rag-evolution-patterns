import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_vectorstore(documents):
    return Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


def load_section_documents(path):
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    documents = []
    for index, block in enumerate(blocks, start=1):
        title = block.splitlines()[0]
        documents.append(Document(page_content=block, metadata={"source": path.name, "section": index, "title": title}))
    return documents


def format_context(documents):
    return "\n\n".join(
        f"[{document.metadata.get('source')}:{document.metadata.get('section')}]\n{document.page_content}"
        for document in documents
    )

require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "unstructured" / "text"
documents = load_section_documents(DATA_DIR / "company_handbook.txt")
vectorstore = build_vectorstore(documents)
llm = make_llm()

chat_history = [
    ("user", "How many days per week can I work from home?"),
    ("assistant", "Employees can work from home up to 3 days a week."),
]
follow_up_query = "What about the core hours for that policy?"

history_text = "\n".join(f"{speaker}: {message}" for speaker, message in chat_history)
standalone_query = llm.invoke(
    "Rewrite the follow-up question into a standalone retrieval query.\n\n"
    f"Chat history:\n{history_text}\n\n"
    f"Follow-up question: {follow_up_query}"
).content.strip()

retrieved_docs = vectorstore.similarity_search(standalone_query, k=2)
context = format_context(retrieved_docs)
answer = llm.invoke(
    "Answer the follow-up question using the retrieved context.\n\n"
    f"Original follow-up: {follow_up_query}\n"
    f"Standalone query: {standalone_query}\n\n"
    f"Context:\n{context}"
).content

print("Follow-up question:", follow_up_query)
print("Standalone query:", standalone_query)
print("\nAnswer:\n", answer)
