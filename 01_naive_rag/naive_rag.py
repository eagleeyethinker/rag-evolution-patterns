import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
HANDBOOK_PATH = BASE_DIR.parent / "data" / "unstructured" / "text" / "company_handbook.txt"

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def load_sections(path):
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    return [
        Document(page_content=block, metadata={"section": index, "source": path.name})
        for index, block in enumerate(blocks, start=1)
    ]


def format_context(documents):
    return "\n\n".join(
        f"[Section {document.metadata['section']}]\n{document.page_content}"
        for document in documents
    )


documents = load_sections(HANDBOOK_PATH)
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query = "What is the remote work policy?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
context = format_context(retrieved_docs)
answer = llm.invoke(
    "Answer the question using the retrieved handbook sections only.\n\n"
    f"Question: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("\nAnswer:\n", answer)
