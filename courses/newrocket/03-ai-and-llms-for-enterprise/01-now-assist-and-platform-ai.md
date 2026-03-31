# 01 – Now Assist and ServiceNow Platform AI

ServiceNow's native AI capabilities are what newRocket builds on top of. Understanding the platform AI layer lets you speak intelligently about what's built-in vs. what newRocket adds.

---

## 1. Now Assist: ServiceNow's GenAI Layer

### What It Is

Now Assist is ServiceNow's generative AI product line — a set of skills (pre-built GenAI capabilities) that plug into existing ServiceNow workflows. It's the fastest-growing product in ServiceNow history: **$600M+ ACV in late 2025, targeting $1B by end of 2026**.

### Now Assist Skills

| Skill | What It Does | Module |
|-------|-------------|--------|
| **Case/Incident Summarization** | Auto-generates summaries from work notes and comments | ITSM, CSM, HRSD |
| **Resolution Notes Generation** | Generates resolution documentation on close | ITSM |
| **Knowledge Article Generation** | Creates KB articles from resolved incidents | Knowledge Management |
| **Virtual Agent Enhancement** | GenAI-powered conversations in Virtual Agent | Self-service |
| **Code Generation** | Text-to-code and text-to-flow | Development |
| **Search Enhancement** | AI-powered search with natural language | Platform-wide |
| **Email Reply Generation** | Draft email responses from context | All modules |
| **Work Notes Generation** | Suggest work notes based on actions taken | ITSM |

### Now LLM: ServiceNow's Domain-Specific Model

ServiceNow doesn't just wrap OpenAI. They built their own LLM:

- **Base:** Partnerships with Hugging Face/StarCoder and NVIDIA
- **Training:** Fine-tuned on ServiceNow-specific data (ITSM terminology, workflow patterns, enterprise language)
- **Architecture:** Uses RAG by default — every generation is grounded in customer data
- **Hosting:** Runs on ServiceNow's infrastructure (customer data doesn't leave the platform)
- **Also supports:** Azure OpenAI integration via Now Assist Skill Kit (NASK) for customers who want GPT-4

**Why this matters:** Enterprise customers care deeply about data sovereignty. ServiceNow hosting their own LLM means customer data stays within ServiceNow's security boundary — a major selling point.

---

## 2. AI Search: The Retrieval Layer

### How It Works

AI Search is ServiceNow's hybrid search engine that combines keyword search, semantic search, and LLM generation.

```
User Query: "How do I reset my VPN password?"

Step 1: KEYWORD SEARCH
  → Traditional text matching across knowledge articles, catalog items, incidents
  → Returns ranked results by TF-IDF / BM25 score

Step 2: SEMANTIC SEARCH (Vector)
  → Query embedded into vector space
  → Cosine similarity against pre-embedded knowledge chunks
  → Returns semantically similar content (even without keyword match)

Step 3: FUSION
  → Combine keyword and semantic scores
  → Reranker model adjusts final ranking
  → Top results become "retrieval context"

Step 4: GENERATION (if enabled)
  → Top retrieved chunks passed to LLM as context
  → LLM generates a synthesized answer
  → Citations linked back to source knowledge articles
```

### Configuring AI Search for RAG

| Setting | What It Controls |
|---------|-----------------|
| **Search sources** | Which tables/knowledge bases to search |
| **Chunking strategy** | How articles are split (~750 words default) |
| **Embedding model** | Which model converts text to vectors |
| **Relevancy threshold** | Minimum score to include in results |
| **Reranker** | Model that combines keyword + semantic scores |
| **Generation model** | Which LLM generates the answer |
| **Citation mode** | How source references appear in the answer |

### Embedding Pipeline

```
Knowledge Article (published)
  → Chunking: split into ~750-word segments
    → Embedding: each chunk → numerical vector (768-1536 dimensions)
      → Storage: vectors stored in ServiceNow's vector database
        → Indexing: HNSW index for fast approximate nearest neighbor search

On query:
  Query text → Embedding → ANN search → Top-K chunks → Reranking → Context for LLM
```

---

## 3. Predictive Intelligence

Before GenAI, ServiceNow had classical ML built in. It's still there and still useful:

| Feature | What It Does | How It Works |
|---------|-------------|-------------|
| **Classification** | Auto-categorize records | Trains on historical data (past incidents → categories) |
| **Similarity** | Find similar records | TF-IDF + cosine similarity on text fields |
| **Clustering** | Group related records | Unsupervised clustering for pattern detection |
| **Regression** | Predict numeric values | Predict resolution time, resource needs |

### When to Use Predictive Intelligence vs. GenAI

| Scenario | Use Predictive Intelligence | Use Now Assist (GenAI) |
|----------|---------------------------|----------------------|
| Categorize incidents | ✓ (simpler, faster, cheaper) | Only if categories are ambiguous or numerous |
| Suggest resolution | | ✓ (needs natural language generation) |
| Find similar incidents | ✓ (well-established) | |
| Generate summaries | | ✓ (inherently a generation task) |
| Route to assignment group | ✓ (classification problem) | |
| Answer user questions | | ✓ (needs comprehension + generation) |
| Predict SLA breach | ✓ (regression model) | |

**As an FDE, knowing when to use classical ML vs. GenAI is a differentiator.** Not everything needs an LLM. Sometimes a simple classifier trained on 10K historical incidents is faster, cheaper, and more reliable.

---

## 4. AI Agent Studio

### What It Is

AI Agent Studio is ServiceNow's low-code tool for building AI agents. It's where newRocket's agents (Phoebe, Ariel, etc.) are configured and deployed.

### Components

```
AI Agent
├── Agent Definition
│   ├── Name, description, persona
│   ├── Assigned topic(s) — what the agent handles
│   └── Guardrails — what the agent can/cannot do
│
├── Topics
│   ├── Topic: "Password Reset"
│   │   ├── Trigger: user mentions password, reset, locked out
│   │   ├── Conversation flow: gather details → verify identity → reset
│   │   └── Fulfillment: call Password Reset catalog item API
│   │
│   └── Topic: "Incident Status"
│       ├── Trigger: user asks about ticket, status, update
│       ├── Conversation flow: get ticket number → lookup → report
│       └── Fulfillment: query incident table, format response
│
├── Knowledge Sources
│   ├── Knowledge bases (RAG retrieval)
│   ├── Catalog items (for fulfillment)
│   └── External sources (via Integration Hub)
│
├── Skills (Now Assist capabilities)
│   ├── Summarization
│   ├── Generation
│   └── Classification
│
└── Governance
    ├── AI Control Tower monitoring
    ├── Decision pathway logging
    └── Human escalation rules
```

### Agent Configuration Options

| Setting | What It Controls |
|---------|-----------------|
| **Persona** | How the agent communicates (formal, friendly, technical) |
| **Confidence threshold** | Below this, agent escalates to human |
| **Allowed actions** | What the agent can do (read-only vs. create/update records) |
| **Escalation rules** | When and how to hand off to human agent |
| **Data access** | Which tables/fields the agent can query |
| **Audit level** | How much of the decision process is logged |

---

## 5. AI Control Tower: Governance

### Why Governance Matters

Enterprise AI needs oversight. AI Control Tower provides:

1. **Decision pathway tracing** — see exactly why the AI made a decision (what data it used, what scores it calculated)
2. **Token consumption visibility** — track costs per agent, per use case, per department
3. **Policy compliance** — enforce rules (e.g., "never auto-close P1 incidents")
4. **Performance monitoring** — accuracy, resolution rate, escalation rate, user satisfaction
5. **Audit trails** — complete record for compliance and regulatory requirements

### Metrics That Matter

| Metric | What It Tells You |
|--------|------------------|
| **Deflection rate** | % of issues resolved without human agent |
| **Auto-resolution rate** | % of tickets the AI fully resolves |
| **Escalation rate** | % of conversations handed to humans |
| **Mean time to resolution (MTTR)** | Comparison: AI-assisted vs. human-only |
| **User satisfaction (CSAT)** | Post-interaction ratings |
| **Confidence distribution** | Are most AI responses high-confidence or borderline? |
| **Token cost per resolution** | Cost efficiency of AI vs. human resolution |
| **False positive rate** | How often does the AI give wrong answers confidently? |

**As an FDE**, you'll configure these dashboards for clients and use the metrics in your Value Realization Dashboard deliverables. Being able to speak to "we reduced MTTR by 40% and achieved 65% deflection rate" is how you demonstrate ROI.
