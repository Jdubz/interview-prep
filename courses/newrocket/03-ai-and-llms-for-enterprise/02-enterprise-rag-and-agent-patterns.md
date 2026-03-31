# 02 – Enterprise RAG and Agent Patterns on ServiceNow

How RAG and multi-agent systems work in the ServiceNow context, and the protocols that enable agent-to-agent collaboration. This is the technical core of what newRocket's Agentic AI team builds.

---

## 1. ServiceNow RAG Pipeline

### End-to-End Architecture

```
DATA INGESTION (offline, batch)
┌───────────────────────────────────────────────────────┐
│ Knowledge Articles                                     │
│ Incident Resolution Notes                              │
│ Change Implementation Plans                            │
│ Problem Root Cause Analysis                            │
│ CMDB CI Documentation                                  │
│ External docs (Confluence, SharePoint — via connector) │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ CHUNKING                          │
│ ├── Split into ~750-word segments │
│ ├── Preserve section boundaries   │
│ ├── Overlap between chunks        │
│ └── Metadata: source, date, type  │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ EMBEDDING                         │
│ ├── Each chunk → dense vector     │
│ ├── 768–1536 dimensions           │
│ └── ServiceNow embedding model    │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ VECTOR STORE                      │
│ ├── HNSW index for ANN search     │
│ ├── Metadata filters              │
│ └── ServiceNow-hosted             │
└───────────────────────────────────┘


QUERY TIME (online, per-request)
┌────────────────────────────────────┐
│ User Query: "VPN not connecting"   │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ QUERY PROCESSING                   │
│ ├── Query embedding (same model)   │
│ ├── Keyword extraction             │
│ └── Query expansion (optional)     │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ HYBRID RETRIEVAL                   │
│ ├── Vector search (semantic)       │
│ ├── Keyword search (BM25)          │
│ ├── Score fusion                   │
│ └── Metadata filtering             │
│     (e.g., only published articles │
│      from last 12 months)          │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ RERANKING                          │
│ ├── Cross-encoder reranker model   │
│ ├── Combines keyword + semantic    │
│ └── Returns top-K chunks (K=3–5)  │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ GENERATION                         │
│ ├── Top chunks → LLM context       │
│ ├── System prompt + user query     │
│ ├── Now LLM or Azure OpenAI        │
│ └── Response with citations        │
└────────────────────────────────────┘
```

### Chunking Strategies for ServiceNow Data

| Data Source | Chunking Approach | Why |
|-------------|------------------|-----|
| Knowledge articles | Section-based + overlap | Articles have clear H2/H3 structure |
| Incident resolution notes | Per-incident (usually short enough) | Context is per-incident |
| CMDB documentation | Per-CI + relationship context | Include connected CIs for context |
| Change plans | Per-phase (plan/build/test/deploy) | Each phase is a distinct context |
| Catalog items | Per-item with variables | Include form fields and fulfillment details |

### Common RAG Failures in Enterprise

| Failure Mode | Symptom | Fix |
|-------------|---------|-----|
| **Stale knowledge** | AI recommends outdated procedures | Automate re-embedding on article publish; filter by last_updated |
| **Chunk boundary splits** | Answer is split across chunks | Increase overlap; use section-aware chunking |
| **Low recall** | AI says "I don't know" for answerable questions | Add more data sources; tune retrieval threshold down |
| **Low precision** | AI retrieves irrelevant chunks | Add metadata filters; tune threshold up; improve reranker |
| **Hallucination** | AI invents steps not in the source | Lower temperature; add "only answer from provided context" instruction |
| **Cross-tenant leakage** | AI shows data from wrong customer/department | Enforce ACLs in retrieval; filter by tenant/department metadata |

---

## 2. Agent Orchestration

### Single Agent vs. Multi-Agent

newRocket's Intelligent Agent Crew uses a **microservices-inspired design**: a core agent supported by specialized helper agents.

```
SINGLE AGENT (Simple)
┌──────────────────────────┐
│ AI Agent: Phoebe         │
│ ├── Topic: Password Reset│
│ ├── Topic: VPN Issues    │
│ ├── Topic: Email Problems │
│ └── Escalation → Human   │
└──────────────────────────┘

MULTI-AGENT (newRocket's approach)
┌──────────────────────────────────────────────────┐
│ Orchestrator Agent                                │
│ ├── Understands user intent                       │
│ ├── Routes to specialist agent                    │
│ ├── Manages conversation state                    │
│ └── Handles cross-agent handoffs                  │
│                                                   │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Phoebe      │ │ Ariel       │ │ Elara       │ │
│ │ (IT Support)│ │ (HR Ops)    │ │ (Knowledge) │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
│ ┌─────────────┐ ┌─────────────┐                  │
│ │ Miles       │ │ Heidi       │                  │
│ │ (IT Ops)    │ │ (Finance)   │                  │
│ └─────────────┘ └─────────────┘                  │
└──────────────────────────────────────────────────┘
```

### Agent-to-Agent Communication

newRocket uses purpose-built protocols for agents to share data, context, and status:

```
User: "My laptop is slow and I need to request a new one"

Orchestrator:
  → Intent 1: Troubleshoot slow laptop → Route to Phoebe
  → Intent 2: Request new laptop → Route to Catalog Agent

Phoebe:
  1. Gather diagnostics (what model, how old, what's slow)
  2. Check CMDB for laptop CI (age, specs, warranty)
  3. Suggest quick fixes from knowledge base
  4. If unfixable → Pass context to Catalog Agent:
     {
       "user": "john.doe",
       "current_ci": "LAPTOP-2024-0456",
       "issue": "performance degradation, 4 years old, out of warranty",
       "recommendation": "replace",
       "diagnostics_completed": true
     }

Catalog Agent:
  1. Receives context from Phoebe (doesn't re-ask questions)
  2. Identifies appropriate catalog item ("Request New Laptop")
  3. Pre-fills form with user info and justification
  4. Routes for manager approval
```

### Orchestration Patterns

| Pattern | How It Works | When to Use |
|---------|-------------|-------------|
| **Sequential** | Agent A completes → passes to Agent B | Multi-step processes with clear handoff points |
| **Parallel** | Agents A and B work simultaneously, results merged | Independent subtasks (e.g., check KB + check CMDB) |
| **Conditional** | Orchestrator picks agent based on intent classification | General-purpose entry point routing |
| **Escalation** | AI agent → human agent with full context | Confidence below threshold, sensitive topics |
| **Feedback loop** | Agent A proposes, Agent B validates | Quality-critical decisions (e.g., auto-approve change requests) |

---

## 3. Agent2Agent (A2A) Protocol

ServiceNow has adopted the **Agent2Agent (A2A) protocol** for inter-agent communication. This is an emerging standard (not just ServiceNow-specific).

### Core Concepts

| Concept | What It Is |
|---------|-----------|
| **Agent Card** | JSON description of an agent's capabilities, inputs, and outputs |
| **Task** | A unit of work sent from one agent to another |
| **Message** | Communication within a task (request, update, completion) |
| **Artifact** | Data produced by the agent (file, record, analysis result) |

### How A2A Fits into ServiceNow

```javascript
// Simplified: Orchestrator sending task to Phoebe
var task = {
    agent: 'phoebe-it-support',
    action: 'troubleshoot',
    input: {
        user_id: 'john.doe',
        issue_description: 'Laptop running slow for the past week',
        ci_sys_id: 'abc123'
    },
    context: {
        conversation_id: 'conv-789',
        priority: 'medium',
        previous_interactions: []
    },
    constraints: {
        max_actions: 5,       // don't take more than 5 actions
        auto_resolve: false,  // suggest, don't auto-fix
        timeout_minutes: 10
    }
};
```

---

## 4. Model Context Protocol (MCP)

### What It Is

MCP (Model Context Protocol) is a standard for giving AI agents access to external tools, data, and systems. ServiceNow has embedded MCP support so agents can reach beyond the ServiceNow platform.

### MCP in the ServiceNow Context

```
AI Agent (on ServiceNow)
  │
  ├── MCP Server: CMDB
  │   → Tool: lookup_ci(name) → returns CI details
  │   → Tool: get_dependencies(ci_id) → returns dependency tree
  │
  ├── MCP Server: Knowledge
  │   → Tool: search_articles(query) → returns relevant articles
  │   → Tool: get_article(id) → returns full article
  │
  ├── MCP Server: External (customer's systems)
  │   → Tool: check_monitoring(server) → returns health status
  │   → Tool: query_logs(service, timerange) → returns log entries
  │
  └── MCP Server: Actions
      → Tool: create_incident(details) → creates ServiceNow incident
      → Tool: update_record(table, id, fields) → updates record
      → Tool: send_notification(user, message) → sends email/slack
```

### Why MCP Matters for FDEs

Every customer has different systems. MCP provides a standardized way to:
1. **Expose customer systems** to AI agents without custom code per system
2. **Control access** — each MCP server declares what tools it offers
3. **Maintain security** — agents can only use tools they're authorized for
4. **Enable reuse** — MCP servers written for one customer can work for another

---

## 5. AI Control Tower: Deep Dive

### Decision Pathway Tracing

Every AI decision is logged with full context:

```
Decision Log Entry:
├── Timestamp: 2026-03-15T14:32:00Z
├── Agent: phoebe-it-support
├── User query: "Can't access email on my phone"
├── Intent classification: email_access_mobile (confidence: 0.92)
├── Retrieval:
│   ├── Query: "mobile email access troubleshoot"
│   ├── Top 3 chunks:
│   │   ├── KB0045123 (score: 0.89) — "Mobile Email Setup Guide"
│   │   ├── KB0045156 (score: 0.84) — "Email Sync Troubleshooting"
│   │   └── INC0067890 (score: 0.78) — Similar resolved incident
│   └── Retrieval latency: 120ms
├── Generation:
│   ├── Model: now-llm-v2
│   ├── Prompt tokens: 1,200
│   ├── Completion tokens: 350
│   ├── Temperature: 0.3
│   └── Response: "Let me help you with mobile email access..."
├── Action taken: Suggested self-service steps
├── User feedback: Resolved (thumbs up)
└── Total cost: $0.003
```

### Guardrail Configuration

```javascript
// Example guardrail rules for an AI agent
{
    "rules": [
        {
            "name": "no_auto_close_p1",
            "condition": "incident.priority == 1",
            "action": "block_auto_resolve",
            "reason": "P1 incidents require human verification"
        },
        {
            "name": "pii_filter",
            "condition": "response contains SSN, credit_card, or password",
            "action": "redact_and_flag",
            "reason": "PII must not appear in AI responses"
        },
        {
            "name": "confidence_floor",
            "condition": "confidence < 0.7",
            "action": "escalate_to_human",
            "reason": "Low-confidence responses should be reviewed"
        },
        {
            "name": "max_actions",
            "condition": "action_count > 3",
            "action": "require_human_approval",
            "reason": "Prevent runaway agent loops"
        }
    ]
}
```

---

## 6. Patterns You'll Implement as an FDE

### Pattern 1: Knowledge-Grounded Resolution

Most common. AI reads knowledge base, suggests resolution.

```
Incident created → AI Search (RAG) → Generate suggestion → Present to agent or user
```

### Pattern 2: Predictive Routing + AI Assist

Combine classical ML (routing) with GenAI (assist).

```
Incident created → ML classifies category → Routes to group → AI generates context summary for assignee
```

### Pattern 3: Conversational Triage

AI agent interviews the user to gather information before creating a ticket.

```
User opens chat → Agent asks clarifying questions → Agent creates fully categorized incident → Routes to right team
```

### Pattern 4: Proactive Ops

AI detects issues before users report them.

```
Event Management alert → Correlate with CMDB → Predict impacted services → Auto-create incident → Suggest remediation
```

### Pattern 5: Knowledge Gap Detection

AI identifies what's missing from the knowledge base.

```
Analyze incidents resolved without KB match → Cluster by topic → Generate draft articles → Queue for review
```

### Pattern 6: Cross-Agent Workflow

Multiple AI agents collaborate on a complex request.

```
User: "I'm starting next Monday"
  → Ariel (HR): Create onboarding case, set up benefits enrollment
  → Phoebe (IT): Provision laptop, email, VPN, badge access
  → Elara (Knowledge): Send welcome docs and orientation materials
  → Orchestrator: Track all tasks, report progress to manager
```
