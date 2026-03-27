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
    return "\n\n".join(
        f"[{document.metadata.get('id')}]\n{document.page_content}"
        for document in documents
    )


def extract_filters(query):
    lower_query = query.lower()
    filters = {}

    if "engineering" in lower_query:
        filters["department"] = "engineering"
    elif "sales" in lower_query:
        filters["department"] = "sales"
    elif "support" in lower_query:
        filters["department"] = "support"

    if "europe" in lower_query or "eu" in lower_query:
        filters["region"] = "eu"
    elif "us" in lower_query or "united states" in lower_query:
        filters["region"] = "us"

    if "manager" in lower_query:
        filters["audience"] = "manager"

    return filters


require_openai_key()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semi_structured"
documents = load_json_documents(DATA_DIR / "policies.json")
llm = make_llm()
query = "What is the home office equipment policy for engineering employees in the EU?"

filters = extract_filters(query)
filtered_documents = [
    document
    for document in documents
    if all(document.metadata.get(key) == value for key, value in filters.items())
]

search_pool = filtered_documents or documents
vectorstore = build_vectorstore(search_pool)
retrieved_docs = vectorstore.similarity_search(query, k=2)

print("Query:", query)
print("Extracted filters:", filters)
print("Documents searched:", len(search_pool))

context = format_context(retrieved_docs)
answer = llm.invoke(
    f"Answer the question using the filtered policy context.\n\nQuestion: {query}\n\nContext:\n{context}"
).content

print("\nAnswer:\n", answer)
