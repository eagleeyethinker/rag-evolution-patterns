import json
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


def require_openai_key():
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")


def make_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_context(documents):
    chunks = []
    for document in documents:
        chunks.append(f"[{document.metadata.get('source')}:{document.metadata.get('id')}]\n{document.page_content}")
    return "\n\n".join(chunks)


def build_graph_context(query):
    graph = json.loads((DATA_DIR / "company_graph.json").read_text(encoding="utf-8"))
    node_lookup = {node["id"]: node for node in graph["nodes"]}
    adjacency = defaultdict(list)

    for edge in graph["edges"]:
        adjacency[edge["source"]].append(edge)
        adjacency[edge["target"]].append(edge)

    matched_nodes = []
    lower_query = query.lower()
    for node in graph["nodes"]:
        if any(token in lower_query for token in node["id"].replace("_", " ").split()) or node["text"].lower() in lower_query:
            matched_nodes.append(node["id"])

    if not matched_nodes:
        matched_nodes = ["home_office_budget", "standing_desks", "product_x"]

    graph_documents = []
    included_nodes = set(matched_nodes)

    for node_id in matched_nodes:
        node = node_lookup[node_id]
        graph_documents.append(
            Document(page_content=node["text"], metadata={"source": "company_graph.json", "id": node_id})
        )
        for edge in adjacency[node_id]:
            included_nodes.add(edge["source"])
            included_nodes.add(edge["target"])
            relation_text = (
                f"{edge['source']} {edge['relation']} {edge['target']}. "
                f"{node_lookup[edge['source']]['text']} {node_lookup[edge['target']]['text']}"
            )
            graph_documents.append(
                Document(
                    page_content=relation_text,
                    metadata={"source": "company_graph.json", "id": f"{edge['source']}-{edge['target']}"},
                )
            )

    for node_id in included_nodes:
        node = node_lookup[node_id]
        graph_documents.append(
            Document(page_content=node["text"], metadata={"source": "company_graph.json", "id": node_id})
        )

    return graph_documents


require_openai_key()

llm = make_llm()
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "semi_structured"
query = "What budget applies to standing desks, and which product has a 3-year warranty?"
graph_context = build_graph_context(query)

print("Query:", query)
print("Graph facts gathered:", len(graph_context))

answer = llm.invoke(
    f"Answer the question using the graph-derived facts.\n\nQuestion: {query}\n\nContext:\n{format_context(graph_context)}"
).content

print("\nAnswer:\n", answer)
