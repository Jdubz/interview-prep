# 02 – ITSM and Workflow Foundations

ServiceNow started as an IT Service Management (ITSM) tool and expanded outward. Understanding ITSM workflows is essential because most AI agent deployments at newRocket automate or augment these existing processes.

---

## 1. ITSM: The Core Workflow Engine

### What ITSM Is

IT Service Management is the practice of designing, delivering, managing, and improving IT services. ServiceNow implements the **ITIL (Information Technology Infrastructure Library)** framework — a set of best practices for IT service delivery.

**Why you care as an FDE:** Every enterprise client has ITSM running on ServiceNow. When you deploy an AI agent like Phoebe (IT support), it's automating steps within these existing ITSM workflows. You need to understand what you're automating.

### The Big 5 ITSM Processes

| Process | What It Does | Key Table | AI Opportunity |
|---------|-------------|-----------|----------------|
| **Incident Management** | Restore service ASAP when something breaks | `incident` | Auto-categorization, resolution suggestion, self-healing |
| **Problem Management** | Find and fix root causes | `problem` | Pattern detection across incidents, root cause analysis |
| **Change Management** | Control and approve planned changes | `change_request` | Risk assessment, impact analysis, auto-approval for low-risk |
| **Service Request Management** | Handle user requests (new laptop, access, etc.) | `sc_request` / `sc_req_item` | Auto-fulfillment, approval routing, conversational catalog |
| **Knowledge Management** | Maintain self-service documentation | `kb_knowledge` | Article generation, gap detection, semantic search |

### Incident Lifecycle (The Workflow You'll Automate Most)

```
1. CREATION
   ├── User reports issue (portal, email, phone, chat)
   ├── Auto-categorization (category, subcategory, assignment group)
   └── Priority calculated (impact × urgency matrix)

2. ASSIGNMENT
   ├── Routed to assignment group based on category
   ├── Auto-assigned to individual (round-robin, skills-based, AI-suggested)
   └── SLA clock starts ticking

3. INVESTIGATION & DIAGNOSIS
   ├── Agent reviews incident, checks knowledge base
   ├── May link to known problems or related incidents
   ├── May escalate (functional or hierarchical)
   └── Updates work notes (internal) and additional comments (customer-facing)

4. RESOLUTION
   ├── Fix applied, resolution notes documented
   ├── State → Resolved
   ├── Customer notified
   └── SLA clock stops

5. CLOSURE
   ├── Customer confirms fix (or auto-close after N days)
   ├── State → Closed
   └── Feeds into reporting and problem management
```

**Where AI fits in every step:**
- **Creation:** AI triages and categorizes automatically, suggests self-service articles before creating a ticket
- **Assignment:** AI routes to the best resolver based on skills and workload
- **Investigation:** AI suggests resolutions from knowledge base and past incidents
- **Resolution:** AI auto-resolves known issues (password resets, access requests)
- **Closure:** AI generates summaries, identifies patterns for problem management

---

## 2. ServiceNow Modules Beyond ITSM

ServiceNow has expanded far beyond IT. newRocket works across all of these:

### IT Operations Management (ITOM)
- **Discovery** — auto-discovers CIs in your infrastructure
- **Event Management** — correlates monitoring alerts into actionable events
- **Cloud Management** — manages cloud resources across AWS/Azure/GCP
- **AI relevance:** Miles (newRocket's IT Operations agent) does predictive AIOps, event correlation, and self-healing here

### IT Business Management (ITBM)
- **Project Portfolio Management** — track and prioritize IT projects
- **Application Portfolio Management** — inventory and rationalize applications
- **Demand Management** — intake and prioritize requests for work

### HR Service Delivery (HRSD)
- **Employee Center** — single portal for all employee needs
- **Case Management** — HR cases (benefits, complaints, accommodations)
- **Lifecycle Events** — onboarding, transfers, offboarding
- **AI relevance:** Ariel (newRocket's HR Operations agent) automates the entire employee lifecycle here

### Customer Service Management (CSM)
- **Case Management** — external customer issues
- **Self-Service Portal** — customer-facing knowledge base and chat
- **AI relevance:** Conversational AI for customer support

### Security Operations (SecOps)
- **Security Incident Response** — manage security incidents
- **Vulnerability Response** — track and remediate vulnerabilities
- **Threat Intelligence** — aggregate and act on threat feeds

---

## 3. Flow Designer: The Visual Workflow Engine

Flow Designer replaced the legacy Workflow Editor and is now the primary way to build automation in ServiceNow. As an FDE, you'll use it constantly.

### Core Concepts

| Concept | What It Is | Example |
|---------|-----------|---------|
| **Flow** | An automated process triggered by an event | "When incident created, auto-categorize with AI" |
| **Trigger** | What starts the flow | Record created, scheduled, REST call, another flow |
| **Action** | A step in the flow | Send email, update record, call API, run script |
| **Subflow** | A reusable set of actions | "Notify assigned group" (called from multiple flows) |
| **Decision** | Conditional branching | "If priority = 1, page on-call" |
| **Spoke** | Pre-built Integration Hub connector | Slack spoke, Azure spoke, ServiceNow spoke |

### Flow Designer vs. Code

```
Flow Designer (visual):
  Trigger: incident.created
  → Action: Look up knowledge articles (AI Search)
  → Decision: Match found with confidence > 0.85?
    → Yes: Auto-resolve, notify user
    → No: Assign to group, start SLA

Equivalent in code (Business Rule + Script Include):
  Much more flexible, but harder to maintain and hand off to customer admins
```

**When to use which:**
- **Flow Designer** when the customer needs to modify the logic after you leave (most of the time)
- **Code** when the logic is too complex for visual tools, or when performance matters

### Integration Hub Spokes

Pre-built connectors that Flow Designer actions can call:

| Spoke | What It Connects |
|-------|-----------------|
| REST | Any REST API |
| SOAP | Legacy SOAP services |
| JDBC | Direct database connections |
| PowerShell | Windows systems |
| SSH | Linux/Unix systems |
| Slack | Slack channels and messages |
| Microsoft Teams | Teams channels and messages |
| AWS / Azure / GCP | Cloud service APIs |
| Jira | Atlassian Jira |
| Salesforce | Salesforce CRM |

---

## 4. SLA Management

### What SLAs Are in ServiceNow

Service Level Agreements define expected response and resolution times. ServiceNow tracks them automatically.

| SLA Type | What It Measures |
|----------|-----------------|
| **Response SLA** | Time from creation to first response |
| **Resolution SLA** | Time from creation to resolution |
| **Assigned SLA** | Time from assignment to resolution |

### Priority Matrix (Standard ITIL)

| Priority | Impact | Urgency | Response | Resolution |
|----------|--------|---------|----------|------------|
| 1 – Critical | High | High | 15 min | 4 hours |
| 2 – High | High | Medium | 30 min | 8 hours |
| 3 – Moderate | Medium | Medium | 4 hours | 2 days |
| 4 – Low | Low | Low | 8 hours | 5 days |

**AI opportunity:** AI agents that auto-resolve P3/P4 incidents free up human agents for P1/P2 work. This is a core value proposition of newRocket's IT support agent Phoebe.

---

## 5. Service Catalog

The Service Catalog is ServiceNow's "app store" for enterprise services. Employees browse a catalog, select items, and submit requests that flow through approval and fulfillment workflows.

### Structure

```
Service Catalog
├── Category: Hardware
│   ├── Catalog Item: Request a Laptop
│   ├── Catalog Item: Request a Monitor
│   └── Catalog Item: Request a Phone
├── Category: Software
│   ├── Catalog Item: Request Software License
│   └── Catalog Item: Request Cloud Access
├── Category: Access
│   ├── Catalog Item: Request VPN Access
│   └── Catalog Item: Request Building Access
└── Category: HR
    ├── Catalog Item: Report a Life Event
    └── Catalog Item: Request Time Off
```

**AI opportunity:** Conversational AI replaces the catalog browsing experience. Instead of navigating categories, the user says "I need access to the production database" and the AI agent identifies the right catalog item, pre-fills the form, and routes for approval.

---

## 6. Knowledge Management

### How It Works

Knowledge articles are stored in `kb_knowledge`, organized into knowledge bases. They go through a lifecycle: Draft → Review → Published → Retired.

### Why It Matters for RAG

Knowledge bases are the primary data source for ServiceNow's RAG implementation:
1. Articles are chunked (~750 words per chunk)
2. Chunks are embedded into vectors
3. When a user asks a question, the query is embedded and matched against article chunks
4. Top matches become context for LLM generation
5. The AI agent synthesizes an answer grounded in knowledge articles

**As an FDE**, you'll frequently help customers:
- Structure their knowledge bases for optimal RAG retrieval
- Identify knowledge gaps (incidents without matching articles)
- Automate knowledge article generation from resolved incidents
