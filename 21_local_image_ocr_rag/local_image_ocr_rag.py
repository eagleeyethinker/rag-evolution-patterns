import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rapidocr_onnxruntime import RapidOCR


def require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_vectorstore(documents):
    return Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


def run_ocr(image_paths):
    ocr = RapidOCR()
    documents = []

    for image_path in image_paths:
        result, _ = ocr(str(image_path))
        if not result:
            continue

        lines = [item[1] for item in result if len(item) >= 2 and item[1].strip()]
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "image_name": image_path.name,
                    "source_type": image_path.suffix.lstrip("."),
                    "line_count": len(lines),
                },
            )
        )

    return documents


def format_context(documents):
    chunks = []
    for document in documents:
        chunks.append(
            f"[{document.metadata['image_name']} | lines={document.metadata['line_count']}]\n{document.page_content}"
        )
    return "\n\n".join(chunks)


require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "unstructured" / "images"
image_paths = [
    DATA_DIR / "sample_invoice.jpg",
    DATA_DIR / "receipt.png",
]

documents = run_ocr(image_paths)
vectorstore = build_vectorstore(documents)
llm = make_llm()

query = "What is the invoice due date, and what total appears on the receipt?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
context = format_context(retrieved_docs)

answer = llm.invoke(
    "Answer the question using only text extracted locally from the images at runtime. "
    "Mention which image each fact came from.\n\n"
    f"Question: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("Images processed:", len(documents))
print("\nExtracted Context Preview:\n", context[:1200])
print("\nAnswer:\n", answer)
