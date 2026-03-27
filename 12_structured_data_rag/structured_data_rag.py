import csv
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


def load_csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_context(documents):
    chunks = []
    for document in documents:
        label = document.metadata.get("product") or document.metadata.get("section")
        chunks.append(f"[{document.metadata.get('source')}:{label}]\n{document.page_content}")
    return "\n\n".join(chunks)

require_openai_key()

BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
handbook_docs = load_section_documents(BASE_DATA_DIR / "unstructured" / "text" / "company_handbook.txt")
catalog_rows = load_csv_rows(BASE_DATA_DIR / "structured" / "equipment_catalog.csv")
catalog_documents = [
    Document(
        page_content=(
            f"{row['product']} is a {row['category']} priced at ${row['price_usd']} "
            f"with a {row['warranty_years']}-year warranty. {row['notes']}"
        ),
        metadata={"source": "equipment_catalog.csv", "product": row["product"]},
    )
    for row in catalog_rows
]

vectorstore = build_vectorstore(handbook_docs + catalog_documents)
llm = make_llm()

query = "Which standing desks are under the home office stipend, and which one has the longest warranty?"
retrieved_docs = vectorstore.similarity_search(query, k=4)

budget = 500
eligible_desks = [row for row in catalog_rows if row["category"] == "standing_desk" and int(row["price_usd"]) <= budget]
structured_summary = "\n".join(
    f"{row['product']}: ${row['price_usd']}, {row['warranty_years']}-year warranty"
    for row in eligible_desks
)

context = format_context(retrieved_docs) + f"\n\n[structured_filter]\n{structured_summary}"
answer = llm.invoke(
    f"Answer the question using the handbook plus the structured table results.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("\nAnswer:\n", answer)
