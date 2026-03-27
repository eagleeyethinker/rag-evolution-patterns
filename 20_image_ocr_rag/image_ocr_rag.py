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


def load_ocr_documents(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for row in rows:
        documents.append(
            Document(
                page_content=row["ocr_text"],
                metadata={
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "image_path": row["image_path"],
                },
            )
        )
    return documents


def format_context(documents):
    chunks = []
    for document in documents:
        image_name = Path(document.metadata["image_path"]).name
        chunks.append(
            f"[{document.metadata['id']} | {document.metadata['source_type']} | {image_name}]\n{document.page_content}"
        )
    return "\n\n".join(chunks)


require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
documents = load_ocr_documents(DATA_DIR / "semi_structured" / "image_ocr_records.json")
vectorstore = build_vectorstore(documents)
llm = make_llm()

query = "Which merchant appears on the receipt, and what amount is shown on the invoice image?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
context = format_context(retrieved_docs)

answer = llm.invoke(
    "Answer the question using OCR text extracted from enterprise images. "
    "Mention which image each fact came from.\n\n"
    f"Question: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("Images indexed:", len(documents))
print("\nAnswer:\n", answer)
