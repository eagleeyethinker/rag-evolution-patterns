import os
import re
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


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_score(query, text):
    query_terms = set(tokenize(query))
    text_terms = tokenize(text)
    overlap = sum(1 for term in text_terms if term in query_terms)
    phrase_bonus = 2 if query.lower() in text.lower() else 0
    return overlap + phrase_bonus


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

query = "How much can I spend on a desk for my WFH setup?"
initial_docs = vectorstore.similarity_search(query, k=2)
initial_score = sum(keyword_score(query, document.page_content) for document in initial_docs)

print("Original query:", query)
print("Initial retrieval score:", initial_score)

working_query = query
final_docs = initial_docs

if initial_score < 8:
    working_query = llm.invoke(
        "Rewrite this question into a clearer retrieval query for an internal policy handbook.\n\n"
        f"Question: {query}"
    ).content.strip()
    final_docs = vectorstore.similarity_search(working_query, k=3)
    print("Corrected query:", working_query)
else:
    print("Correction not needed.")

context = format_context(final_docs)
answer = llm.invoke(
    f"Answer the original question using the corrected retrieval context.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("\nAnswer:\n", answer)
