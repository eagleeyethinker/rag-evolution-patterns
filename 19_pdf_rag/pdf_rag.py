import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_vectorstore(documents):
    return Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


def format_context(documents):
    chunks = []
    for document in documents:
        source = Path(document.metadata.get("source", "unknown")).name
        page = document.metadata.get("page", 0) + 1
        chunks.append(f"[{source}:page-{page}]\n{document.page_content}")
    return "\n\n".join(chunks)


require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "unstructured" / "pdfs"
pdf_paths = [
    DATA_DIR / "enterprise_invoice_sample.pdf",
    DATA_DIR / "irs_1040_sample.pdf",
]

documents = []
for pdf_path in pdf_paths:
    documents.extend(PyPDFLoader(str(pdf_path)).load())

vectorstore = build_vectorstore(documents)
llm = make_llm()

query = "What is the invoice number, due date, and total charge on the enterprise invoice?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
context = format_context(retrieved_docs)

answer = llm.invoke(
    "Answer the question using the PDF pages only. "
    "Cite the page labels exactly as shown in brackets.\n\n"
    f"Question: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("PDF files loaded:", len(pdf_paths))
print("\nAnswer:\n", answer)
