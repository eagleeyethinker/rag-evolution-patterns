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
documents = load_section_documents(DATA_DIR / "company_handbook.txt") + load_section_documents(DATA_DIR / "workspace_guides.txt")
vectorstore = build_vectorstore(documents)
llm = make_llm()

query = "What warranty applies to the company laptop?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
compressed_docs = []

for document in retrieved_docs:
    compressed_text = llm.invoke(
        "Extract only the lines that help answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Document:\n{document.page_content}"
    ).content
    compressed_docs.append(document.model_copy(update={"page_content": compressed_text}))

print("Query:", query)
print("Retrieved docs:", len(retrieved_docs))
print("Compressed docs:", len(compressed_docs))

context = format_context(compressed_docs)
answer = llm.invoke(
    f"Answer the question using the compressed context only.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("\nAnswer:\n", answer)
