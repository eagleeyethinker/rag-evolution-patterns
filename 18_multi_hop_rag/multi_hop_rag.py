import csv
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


def require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


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

require_openai_key()

llm = make_llm()
BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
handbook_docs = load_section_documents(BASE_DATA_DIR / "unstructured" / "text" / "company_handbook.txt")
catalog_rows = load_csv_rows(BASE_DATA_DIR / "structured" / "equipment_catalog.csv")

query = "What is the home office budget, and which standing desk under that amount has the longest warranty?"

budget_doc = next(document for document in handbook_docs if "stipend" in document.page_content.lower())
budget = 500

eligible_desks = [row for row in catalog_rows if row["category"] == "standing_desk" and int(row["price_usd"]) <= budget]
best_desk = max(eligible_desks, key=lambda row: int(row["warranty_years"]))

hop_context = "\n".join(
    [
        f"Hop 1: {budget_doc.page_content}",
        "Hop 2: Eligible desks under the stipend:",
        *[
            f"- {row['product']} costs ${row['price_usd']} with a {row['warranty_years']}-year warranty."
            for row in eligible_desks
        ],
    ]
)

answer = llm.invoke(
    "Answer the question using the multi-hop reasoning trace.\n\n"
    f"Question: {query}\n\n"
    f"Reasoning trace:\n{hop_context}\n\n"
    f"Best match after hop 2: {best_desk['product']}"
).content

print("Query:", query)
print("\nAnswer:\n", answer)
