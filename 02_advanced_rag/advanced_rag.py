import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

BASE_DIR = Path(__file__).resolve().parent
HANDBOOK_PATH = BASE_DIR.parent / "data" / "unstructured" / "text" / "company_handbook.txt"

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def load_sections(path):
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    return [
        Document(page_content=block, metadata={"section": index, "source": path.name})
        for index, block in enumerate(blocks, start=1)
    ]


documents = load_sections(HANDBOOK_PATH)
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
)

query = "Tell me about WFH rules."
docs = advanced_retriever.invoke(query)

print("Original Query:", query)
print(f"Retrieved {len(docs)} documents using query expansion.")
print("\nExpanded Context:\n")
for document in docs:
    print(f"[Section {document.metadata.get('section')}]\n{document.page_content}\n")
