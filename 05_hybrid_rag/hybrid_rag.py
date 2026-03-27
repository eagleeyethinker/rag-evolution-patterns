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


def keyword_search(query, documents, k=3):
    ranked = sorted(documents, key=lambda doc: keyword_score(query, doc.page_content), reverse=True)
    return [doc for doc in ranked[:k] if keyword_score(query, doc.page_content) > 0]


def merge_unique_documents(*document_lists):
    unique_documents = []
    seen = set()
    for document_list in document_lists:
        for document in document_list:
            identity = (document.metadata.get("source"), document.metadata.get("section"), document.page_content)
            if identity not in seen:
                seen.add(identity)
                unique_documents.append(document)
    return unique_documents


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

query = "What are the work-from-home rules and what budget can I use for a standing desk?"

dense_results = vectorstore.similarity_search(query, k=2)
keyword_results = keyword_search(query, documents, k=2)
hybrid_results = merge_unique_documents(dense_results, keyword_results)

print("Query:", query)
print("Dense hits:", len(dense_results))
print("Keyword hits:", len(keyword_results))
print("Hybrid hits:", len(hybrid_results))

context = format_context(hybrid_results)
answer = llm.invoke(
    f"Answer the question using the combined retrieval context.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("\nAnswer:\n", answer)
