# 01 – newRocket's Intelligent Agent Crew

newRocket's flagship AI product. Launched October 2025 — a catalog of 9 purpose-built AI agents on ServiceNow. Understanding these agents inside-out shows the interviewer you've done your homework and can speak to the product you'd be deploying.

---

## 1. Overview

The Intelligent Agent Crew is a catalog of **enterprise-grade, modular, customizable AI agents** built on the ServiceNow platform. Each agent targets a specific enterprise workflow domain.

### Design Principles

| Principle | What It Means |
|-----------|--------------|
| **Purpose-built** | Each agent solves a specific domain problem, not a general chatbot |
| **Modular** | Agents can be deployed independently or composed together |
| **Customizable** | Customers tailor agents to their specific workflows and data |
| **Enterprise-grade** | Security, governance, audit trails, compliance-ready |
| **Measurable** | Built-in metrics tied to business value (not just AI accuracy) |

### Architecture: Microservices-Inspired

Each agent follows a **core + helper** pattern:

```
Agent: Phoebe (IT Support)
├── Core Agent
│   ├── Intent classification
│   ├── Conversation management
│   ├── Response generation
│   └── Action execution
│
├── Helper: Knowledge Retriever
│   ├── RAG pipeline
│   ├── Article scoring
│   └── Citation generation
│
├── Helper: CMDB Enricher
│   ├── CI lookup
│   ├── Dependency mapping
│   └── Impact assessment
│
├── Helper: Ticket Manager
│   ├── Create/update incidents
│   ├── SLA monitoring
│   └── Assignment routing
│
└── Helper: Escalation Handler
    ├── Confidence monitoring
    ├── Human handoff
    └── Context transfer
```

**Agent-to-Agent collaboration** uses purpose-built protocols for sharing data, context, and status in real time. The orchestration layer handles sequencing, conditional decisions, and task handoffs.

---

## 2. The Nine Agents

### Launched (Available Now)

#### Phoebe — IT Support Agent
- **Domain:** IT Service Management
- **What it does:** Intelligent IT support entry point. Resolves common issues (password resets, VPN, email) and handles escalations with full context.
- **Key capabilities:**
  - Conversational triage (replaces static forms)
  - Knowledge-grounded resolution suggestions
  - Auto-creation of fully categorized incidents
  - Intelligent routing based on skills and workload
  - Seamless escalation to human agents with conversation history

**FDE deployment tasks:** Configure knowledge sources, customize triage flows, integrate with customer's monitoring tools, set confidence thresholds, tune RAG retrieval for customer's KB.

#### Ariel — HR Operations Agent
- **Domain:** HR Service Delivery
- **What it does:** Automates the complete employee lifecycle from onboarding through offboarding.
- **Key capabilities:**
  - Onboarding workflow automation (provisioning, documentation, training)
  - Benefits enrollment assistance
  - Policy and procedure Q&A (grounded in HR knowledge base)
  - Life event processing (marriage, new child, relocation)
  - Offboarding coordination (access revocation, equipment return, exit interview)

**FDE deployment tasks:** Map customer's HR policies into knowledge base, configure lifecycle event workflows, integrate with HRIS (Workday, SAP SuccessFactors), customize approval chains.

#### Elara — Knowledge Agent
- **Domain:** Knowledge Management (cross-platform)
- **What it does:** Intelligent knowledge retrieval and enrichment across ServiceNow and external systems.
- **Key capabilities:**
  - Cross-platform knowledge search (ServiceNow KB + Confluence + SharePoint + custom)
  - Knowledge gap identification
  - Article quality scoring
  - Automated article generation from resolved tickets
  - Multi-language support

**FDE deployment tasks:** Connect external knowledge sources, configure cross-platform search, set up article generation pipelines, tune retrieval quality.

#### Miles — IT Operations Agent
- **Domain:** IT Operations Management (ITOM)
- **What it does:** Predictive AIOps, event correlation, and self-healing.
- **Key capabilities:**
  - Alert correlation (reduce noise from 1000s of alerts to actionable events)
  - Predictive incident detection (spot problems before users report them)
  - Automated remediation (self-healing for known patterns)
  - Impact analysis using CMDB dependency graphs
  - Capacity planning recommendations

**FDE deployment tasks:** Integrate with monitoring tools (Datadog, Splunk, PagerDuty, CloudWatch), configure correlation rules, build remediation playbooks, map CMDB dependencies.

#### Heidi — Financial Operations Agent
- **Domain:** Finance and Administration
- **What it does:** Case management, compliance workflows, treasury optimization.
- **Key capabilities:**
  - Financial case management automation
  - Compliance workflow enforcement
  - Invoice processing and matching
  - Budget tracking and anomaly detection
  - Audit preparation assistance

**FDE deployment tasks:** Map customer's financial workflows, integrate with ERP (SAP, Oracle, NetSuite), configure compliance rules, build approval workflows.

### Coming Soon

| Agent | Domain | Focus |
|-------|--------|-------|
| **Security Operations** | SecOps | Incident response, vulnerability triage, threat intelligence |
| **Managed Services** | MSP/MSSP | Multi-tenant service delivery, SLA management |
| **Platform Automation** | Platform Admin | Instance health, upgrade readiness, configuration optimization |
| **Telecom Operations** | Telecom | Network management, service assurance, customer operations |

---

## 3. How Agents Get Deployed (Your Job)

### The Deployment Lifecycle

```
Phase 1: ASSESSMENT
├── Audit customer's current ServiceNow environment
├── Identify which agents apply (not every customer needs all 9)
├── Assess data quality (knowledge base, CMDB, historical tickets)
├── Identify integration requirements (external systems)
└── Define success metrics with customer

Phase 2: CONFIGURATION
├── Install agent scoped application
├── Configure agent parameters (confidence thresholds, persona, guardrails)
├── Connect knowledge sources (ServiceNow KB + external)
├── Set up integrations (monitoring, HRIS, ERP, collaboration tools)
├── Configure AI Control Tower governance
└── Build custom topics/flows for customer-specific use cases

Phase 3: CUSTOMIZATION
├── Train/fine-tune classification models on customer's historical data
├── Build custom helper agents for customer-specific workflows
├── Create custom spoke actions for unique integrations
├── Develop customer-specific RAG pipelines
└── Build value dashboards

Phase 4: TESTING & VALIDATION
├── UAT with customer stakeholders
├── A/B testing (AI-assisted vs. traditional)
├── Edge case testing and guardrail validation
├── Performance/load testing
└── Security review

Phase 5: GO-LIVE & OPTIMIZATION
├── Gradual rollout (pilot group → department → organization)
├── Monitor metrics via Value Realization Dashboard
├── Tune based on real usage data
├── Weekly check-ins with customer during stabilization
└── Document reusable patterns for newRocket platform
```

### Customization Depth Spectrum

```
Low customization ◄─────────────────────────────► High customization

Install & configure    Modify topics/flows    Build custom agents
(~1 week)              (~2-3 weeks)           (~4-8 weeks)

Standard use cases     Customer-specific       Novel workflows,
with minor tweaks      processes, custom       custom integrations,
                       integrations            new agent types
```

---

## 4. Competitive Context

### How newRocket's Agents Compare

| Aspect | ServiceNow Native (Now Assist) | newRocket Intelligent Agent Crew | Big 4 Custom AI |
|--------|-------------------------------|----------------------------------|-----------------|
| **Speed to deploy** | Fast (built-in) | Fast (pre-built + customize) | Slow (built from scratch) |
| **Depth** | General-purpose skills | Domain-specific agents | Custom to requirement |
| **Customization** | Limited | Moderate (modular design) | Unlimited |
| **Cost** | Included in license | Service + license | Very expensive |
| **Governance** | AI Control Tower | AI Control Tower + AI Traffic Controller | Custom or none |
| **Reusability** | High (platform feature) | High (agent catalog) | Low (bespoke per client) |

### newRocket's Value Add Over Native

1. **Pre-built domain expertise** — Agents come with domain-specific topics, flows, and best practices baked in
2. **Faster deployment** — Weeks instead of months to production
3. **Cross-platform knowledge** — Elara reaches beyond ServiceNow (Confluence, SharePoint, etc.)
4. **Value measurement** — Built-in ROI dashboards from day one
5. **Forward deployed engineering** — Your role: expert humans + pre-built agents = fast, customized outcomes

---

## 5. Questions You Should Be Able to Answer

1. **"Walk me through how you'd deploy Phoebe for a new customer"** — Assessment → Configuration → Customization → Testing → Go-live (see lifecycle above)
2. **"How do the agents communicate with each other?"** — A2A protocol, orchestration layer handles sequencing and context sharing
3. **"What makes these agents enterprise-grade?"** — Governance (AI Control Tower), audit trails, security (ACLs, data isolation), scalability (ServiceNow platform SLA)
4. **"How would you handle a customer whose knowledge base is poor quality?"** — Knowledge gap analysis with Elara, auto-generation of articles from resolved incidents, quality scoring, incremental improvement plan
5. **"Why not just use Now Assist out of the box?"** — Now Assist provides skills; newRocket's agents provide complete domain solutions with orchestration, customization, and value measurement built in
