import json
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


def load_json_documents(path, content_field="content"):
    rows = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for row in rows:
        metadata = {key: value for key, value in row.items() if key != content_field}
        documents.append(Document(page_content=row[content_field], metadata=metadata))
    return documents


def format_context(documents):
    chunks = []
    for document in documents:
        chunks.append(f"[{document.metadata.get('id')}:{document.metadata.get('modality')}]\n{document.page_content}")
    return "\n\n".join(chunks)

require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semi_structured"
documents = load_json_documents(DATA_DIR / "multimodal_records.json")
vectorstore = build_vectorstore(documents)
llm = make_llm()

query = "What does the whiteboard say about using the stipend for desks?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
context = format_context(retrieved_docs)

answer = llm.invoke(
    "Answer the question using multimodal retrieval context. "
    "The documents are text extracted from slides, images, diagrams, and tables.\n\n"
    f"Question: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("\nAnswer:\n", answer)
