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


documents = load_sections(HANDBOOK_PATH)
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def multi_step_reasoning(complex_query):
    print(f"Original Goal: {complex_query}\n")

    decomposition = llm.invoke(
        "Break this question into exactly 2 short sub-questions, one per line, with no numbering.\n\n"
        f"Question: {complex_query}"
    ).content
    sub_queries = [line.strip("- ").strip() for line in decomposition.splitlines() if line.strip()]

    context_chunks = []
    for query in sub_queries:
        print(f"-> Retrieving for: {query}")
        docs = vectorstore.similarity_search(query, k=1)
        if docs:
            doc = docs[0]
            context_chunks.append(f"[Section {doc.metadata['section']}]\n{doc.page_content}")

    context = "\n\n".join(context_chunks)
    return llm.invoke(
        "Answer the original question using the retrieved context.\n\n"
        f"Question: {complex_query}\n\nContext:\n{context}"
    ).content


query = "What is the home office budget, and what are the core hours for remote work?"
print("\nFinal Answer:\n", multi_step_reasoning(query))
