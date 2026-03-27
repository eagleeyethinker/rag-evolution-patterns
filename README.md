# The Evolution of RAG Patterns

A companion repository showing how retrieval-augmented generation evolves from a simple vector lookup into more capable retrieval systems.

## Featured Article

Read the full long-form guide here:

- [The Ultimate Guide to RAG Patterns](docs/ultimate-guide-to-rag-patterns.md)

It covers every implemented pattern in this repo, groups them into practical categories, includes Mermaid diagrams, and links directly to the code for each pattern.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set your OpenAI API key in your shell.
4. Run the demos in order.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
python .\01_naive_rag\naive_rag.py
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python 01_naive_rag/naive_rag.py
```

## Demo Sequence
1. `01_naive_rag`: basic vector retrieval plus answer synthesis.
2. `02_advanced_rag`: query expansion with a multi-query retriever.
3. `03_multi_step_rag`: decomposes a complex question into sub-questions.
4. `04_agentic_rag`: lets an agent choose between internal retrieval and web search.
5. `05_hybrid_rag`: combines dense retrieval with keyword search.
6. `06_reranked_rag`: retrieves a broader candidate set, then reranks it.
7. `07_metadata_filtered_rag`: applies structured filters before semantic retrieval.
8. `08_parent_document_rag`: retrieves child chunks but returns full parent sections.
9. `09_contextual_compression_rag`: compresses retrieved documents before answering.
10. `10_corrective_rag`: retries retrieval after rewriting a weak query.
11. `11_graph_rag`: traverses graph-shaped facts instead of only chunk similarity.
12. `12_structured_data_rag`: augments text retrieval with structured table data.
13. `13_conversational_rag`: rewrites follow-up questions using chat history.
14. `14_citation_grounded_rag`: answers with explicit source references.
15. `15_adaptive_router_rag`: routes each query to the most relevant retriever.
16. `16_multimodal_rag`: retrieves from text extracted out of slides, images, and diagrams.
17. `17_fusion_rag`: fuses rankings from multiple retrievers and query variants.
18. `18_multi_hop_rag`: performs chained retrieval across multiple hops.
19. `19_pdf_rag`: retrieves and cites information from realistic invoice and tax-form PDFs.
20. `20_image_ocr_rag`: retrieves from OCR-extracted text tied to realistic invoice and receipt images.
21. `21_local_image_ocr_rag`: performs OCR locally on real document images before retrieval.

## Data Layout
- `data/structured/equipment_catalog.csv`: structured product table used by mixed retrieval demos.
- `data/semi_structured/policies.json`: metadata-rich policy records.
- `data/semi_structured/company_graph.json`: graph facts for graph-based retrieval.
- `data/semi_structured/multimodal_records.json`: extracted records from slides, images, diagrams, and tables.
- `data/semi_structured/image_ocr_records.json`: OCR output linked to the enterprise images.
- `data/semi_structured/receipt-result.json`: extracted receipt analysis result.
- `data/unstructured/text/company_handbook.txt`: core handbook used by the first demos.
- `data/unstructured/text/workspace_guides.txt`: extra policy snippets for retrieval quality experiments.
- `data/unstructured/text/employee_playbook.txt`: larger sections for parent-child retrieval.
- `data/unstructured/pdfs/enterprise_invoice_sample.pdf`: real-looking invoice PDF sample.
- `data/unstructured/pdfs/irs_1040_sample.pdf`: official-looking tax-form PDF sample.
- `data/unstructured/images/sample_invoice.jpg`: real-looking invoice image sample.
- `data/unstructured/images/receipt.png`: real-looking scanned receipt sample.
- `data/unstructured/images/contract.png`: real-looking contract image sample.
