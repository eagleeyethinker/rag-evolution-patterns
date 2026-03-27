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


def route_query(query):
    lower_query = query.lower()
    policy_terms = any(term in lower_query for term in ["policy", "remote", "hours", "stipend", "budget"])
    equipment_terms = any(term in lower_query for term in ["desk", "chair", "monitor", "warranty", "price"])

    if policy_terms and equipment_terms:
        return "hybrid"
    if policy_terms:
        return "internal_policy"
    if equipment_terms:
        return "equipment_catalog"
    return "hybrid"


require_openai_key()

BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
handbook_store = build_vectorstore(load_section_documents(BASE_DATA_DIR / "unstructured" / "text" / "company_handbook.txt"))
catalog_rows = load_csv_rows(BASE_DATA_DIR / "structured" / "equipment_catalog.csv")
catalog_documents = [
    Document(
        page_content=(
            f"{row['product']} is a {row['category']} priced at ${row['price_usd']} "
            f"with a {row['warranty_years']}-year warranty."
        ),
        metadata={"source": "equipment_catalog.csv", "product": row["product"]},
    )
    for row in catalog_rows
]
catalog_store = build_vectorstore(catalog_documents)
llm = make_llm()

query = "Which standing desk under our stipend has the longest warranty?"
route = route_query(query)

if route == "internal_policy":
    retrieved_docs = handbook_store.similarity_search(query, k=2)
elif route == "equipment_catalog":
    retrieved_docs = catalog_store.similarity_search(query, k=3)
else:
    retrieved_docs = handbook_store.similarity_search(query, k=2) + catalog_store.similarity_search(query, k=2)

context = format_context(retrieved_docs)
answer = llm.invoke(
    f"Answer the question using the routed retrieval context.\n\nRoute: {route}\nQuestion: {query}\n\nContext:\n{context}"
).content

print("Query:", query)
print("Chosen route:", route)
print("\nAnswer:\n", answer)
