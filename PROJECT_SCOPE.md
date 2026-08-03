# SEC 8-K Event Intelligence Copilot

## 1. Business Objective

Build a multi-agent research copilot that helps analysts discover,
understand, and compare material corporate events disclosed through
SEC 8-K filings.

The system retrieves relevant filings, extracts normalized corporate
events, verifies generated claims against filing evidence, and produces
cited timelines and comparisons.

### Initial users

- Equity analysts
- Risk analysts
- Corporate strategy teams
- Investment researchers

### MVP questions

- What major events did a company disclose during a given period?
- Show leadership changes for a company.
- Find cybersecurity incidents across companies.
- Compare restructuring events between companies.
- Generate a chronological event timeline.

### Boundaries

The system is a research and decision-support tool. It does not provide
investment advice, legal conclusions, stock predictions, or autonomous
trading actions.

## 2. Business Outcome

Reduce the manual effort required to identify and review relevant 8-K
filings while producing consistent, structured, and source-verifiable
event intelligence.

### Expected outcomes

- Faster initial filing review
- Improved event discovery
- Consistent event classification
- Evidence-backed summaries
- Reduced unsupported claims
- Reusable normalized event records

### Initial success metrics

- Event classification accuracy
- Event extraction precision, recall, and F1
- Retrieval Recall@K
- Citation correctness
- Grounded-claim rate
- Analyst acceptance rate
- Review-time reduction
- End-to-end latency
- Cost per successful query

## 3. Target Architecture

The MVP uses a controlled LangGraph workflow consisting of:

1. Query Planning Agent
2. Filing Retrieval Agent
3. Event Extraction Agent
4. Verification Agent
5. Response Generator

### Data and model stack

- SEC 8-K dataset stored in Parquet
- Qdrant for vectors and metadata payloads
- Ollama for local model inference
- nomic-embed-text for embeddings
- qwen2.5:7b-instruct for initial agents
- Pydantic for structured contracts
- FastAPI for the service layer
- Streamlit for the demonstration UI
- OpenTelemetry and Jaeger for tracing

### Workflow

START
→ Query Planning
→ Metadata Filtering
→ Semantic Retrieval
→ Event Extraction
→ Verification

Verification routes to one of:

- Response generation
- Retrieval retry
- Human review
- Controlled failure

Retrieval retries are bounded to two iterations.