from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from pathlib import Path


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


def split_parent_into_children(parent_documents):
    children = []
    for parent in parent_documents:
        lines = [line.strip() for line in parent.page_content.splitlines() if line.strip()]
        title = lines[0]
        for index, line in enumerate(lines[1:], start=1):
            children.append(
                Document(
                    page_content=f"{title}: {line}",
                    metadata={
                        "source": parent.metadata["source"],
                        "parent_section": parent.metadata["section"],
                        "child_section": index,
                    },
                )
            )
    return children


require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "unstructured" / "text"
parent_documents = load_section_documents(DATA_DIR / "employee_playbook.txt")
child_documents = split_parent_into_children(parent_documents)
vectorstore = build_vectorstore(child_documents)
llm = make_llm()

query = "What are the eligible purchases for the home office stipend?"
child_hits = vectorstore.similarity_search(query, k=2)
matched_parent_ids = {document.metadata["parent_section"] for document in child_hits}
parent_hits = [document for document in parent_documents if document.metadata["section"] in matched_parent_ids]

print("Query:", query)
print("Child hits:", len(child_hits))
print("Expanded parent docs:", len(parent_hits))

context = format_context(parent_hits)
answer = llm.invoke(
    f"Answer the question using the full parent sections returned from the child hits.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("\nAnswer:\n", answer)
