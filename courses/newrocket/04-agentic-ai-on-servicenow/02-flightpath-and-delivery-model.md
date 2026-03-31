# 02 – FlightPath.AI and newRocket's Delivery Model

newRocket's methodology for delivering AI value. As an FDE, FlightPath.AI is the framework you'll operate within on every engagement.

---

## 1. FlightPath.AI

### What It Is

FlightPath.AI is newRocket's **4-week accelerator** for taking enterprises from AI readiness assessment to working prototype. Launched at ServiceNow Knowledge 2025 (Las Vegas, 20,000+ attendees).

### The 4-Week Sprint

```
WEEK 1: DISCOVER
┌──────────────────────────────────────────────────────────┐
│ Activities:                                               │
│ ├── Align priorities with executive sponsors              │
│ ├── Identify pain points and current workflow bottlenecks │
│ ├── Explore AI opportunities (which agents, which data)   │
│ ├── Assess AI readiness (data quality, infrastructure)    │
│ └── Define success metrics and value benchmarks           │
│                                                           │
│ Deliverables:                                             │
│ ├── AI opportunity map (prioritized by impact + effort)   │
│ ├── Data readiness assessment                             │
│ └── Engagement scope and success criteria                 │
└──────────────────────────────────────────────────────────┘

WEEKS 2-3: PROTOTYPE
┌──────────────────────────────────────────────────────────┐
│ Activities:                                               │
│ ├── Co-develop functional solutions with real data        │
│ ├── Configure and customize selected agents               │
│ ├── Build integrations with customer systems              │
│ ├── Implement RAG pipelines on customer's knowledge       │
│ ├── Iterate based on stakeholder feedback                 │
│ └── Daily/frequent demos and check-ins                    │
│                                                           │
│ Deliverables:                                             │
│ ├── Working prototype in customer's ServiceNow instance   │
│ ├── Integration with at least one external system         │
│ └── Initial performance metrics                           │
└──────────────────────────────────────────────────────────┘

WEEK 4: DEMONSTRATE
┌──────────────────────────────────────────────────────────┐
│ Activities:                                               │
│ ├── Showcase results to stakeholders and sponsors         │
│ ├── Present impact benchmarks and value metrics           │
│ ├── Define governance framework                           │
│ └── Create tailored roadmap to production scale           │
│                                                           │
│ Deliverables:                                             │
│ ├── Working prototype demonstration                       │
│ ├── Value realization benchmarks                          │
│ ├── Governance framework                                  │
│ └── Tailored roadmap for production deployment            │
└──────────────────────────────────────────────────────────┘
```

### Why 4 Weeks (Not 4 Months)

The entire point of FlightPath.AI is to **compress the time-to-value**. The industry problem newRocket is solving:

> 2/3 of ServiceNow implementations stall after the first wave — not due to the platform, but because delivery models haven't adapted to AI-first.
> — HFS Research / newRocket, December 2025

Traditional approach: 3-6 month discovery → 6-12 month implementation → maybe AI features in year 2.

FlightPath.AI: Working AI prototype in 4 weeks → production scale decision with real data.

### Your Role in FlightPath.AI

As the FDE, you are the **technical engine** of the sprint:

| Week | Your Focus |
|------|-----------|
| **Week 1** | Technical discovery: audit current ServiceNow setup, assess data quality, identify integration complexity, sketch architecture |
| **Weeks 2-3** | Heads-down building: configure agents, write integrations, build RAG pipelines, iterate on feedback daily |
| **Week 4** | Demo preparation, metrics collection, architecture documentation, handoff materials, product feedback to newRocket |

---

## 2. Value Realization Dashboard

### What It Is

A dashboard that connects every AI initiative to **measurable business impact**. This is how newRocket proves ROI to customers (and to their own PE owners).

### Key Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Efficiency** | Mean time to resolution (MTTR), first contact resolution rate, tickets per agent |
| **Deflection** | Self-service resolution rate, chatbot deflection rate, auto-resolution rate |
| **Cost** | Cost per ticket (AI-assisted vs. manual), token consumption, FTE hours saved |
| **Quality** | Customer satisfaction (CSAT), first-time fix rate, SLA compliance rate |
| **Adoption** | AI usage rate, feature utilization, user engagement trends |
| **Value** | Estimated $ saved, estimated FTE equivalent freed, projected annual impact |

### Token Consumption Visibility

The dashboard tracks AI cost at a granular level:

```
Token Usage Report:
├── By Agent: Phoebe: 2.1M tokens, Ariel: 800K tokens, Elara: 1.5M tokens
├── By Use Case: Incident resolution: 60%, Knowledge search: 25%, Other: 15%
├── By Model: Now LLM: 70%, Azure OpenAI GPT-4: 30%
├── Cost: $450/month (vs. estimated $12,000/month in analyst hours saved)
└── Trend: 15% reduction in tokens/resolution over 4 weeks (quality improving)
```

**Why this matters:** Enterprise customers need to justify AI spend. The Value Realization Dashboard gives them the numbers to take to their CFO. As an FDE, you'll configure these dashboards and present the findings.

---

## 3. AI Traffic Controller

### What It Is

newRocket's framework for monitoring and controlling agent behavior. Think of it as the "ops layer" on top of AI Control Tower.

### Capabilities

| Feature | What It Does |
|---------|-------------|
| **Behavioral monitoring** | Track what agents are doing in real-time |
| **Configurable guardrails** | Set rules for what agents can/cannot do |
| **Audit trails** | Complete record of every AI decision |
| **Alert system** | Notify admins when agents behave unexpectedly |
| **Kill switch** | Disable an agent immediately if needed |
| **A/B testing** | Compare different agent configurations |

### Guardrail Examples

```
Rule: "Don't auto-resolve after business hours"
Condition: time.now() > 18:00 OR time.now() < 08:00
Action: Queue for next-business-day review instead of auto-resolve
Reason: Customer's policy requires human oversight for off-hours resolutions

Rule: "Escalate financial requests above threshold"
Condition: request.estimated_cost > $5,000
Action: Route to manager approval, disable auto-fulfillment
Reason: Compliance requirement for expenditure authorization

Rule: "Rate limit per user"
Condition: user.interactions_last_hour > 20
Action: Slow down responses, suggest human agent
Reason: Prevent abuse and ensure fair resource allocation
```

---

## 4. Data Intelligence Platform

### What It Is

newRocket's platform for managing the data foundation that AI agents depend on. Poor data → poor AI. This platform ensures data quality.

### Key Functions

| Function | What It Does |
|----------|-------------|
| **Data quality scoring** | Rate the completeness, accuracy, and freshness of data sources |
| **Knowledge gap analysis** | Identify topics with incidents but no knowledge articles |
| **CMDB health assessment** | Score CMDB completeness and relationship accuracy |
| **Data preparation** | Clean, normalize, and enrich data for AI consumption |
| **Continuous monitoring** | Track data quality over time, alert on degradation |

### Why It Matters

```
Bad data scenario:
  CMDB is 60% complete → Miles can't accurately map dependencies
  Knowledge base hasn't been updated in 18 months → Phoebe suggests outdated procedures
  Incident categorization is inconsistent → Classification model trains on noisy labels

Good data scenario:
  CMDB is 95% complete with automated discovery → Miles accurately predicts impact
  Knowledge base refreshed weekly with auto-generated articles → Phoebe resolves 70% of tickets
  Consistent categorization enforced by AI + human review → Classification accuracy >90%
```

**As an FDE**, data quality assessment is often your first task in a new engagement. If the data isn't ready, the AI won't work, and you need to be upfront about that with the customer.

---

## 5. Now Assist LAUNCH

### What It Is

A **10-week program** to embed up to four out-of-the-box Now Assist GenAI skills into a customer's ServiceNow environment.

### How It Differs from FlightPath.AI

| Aspect | FlightPath.AI | Now Assist LAUNCH |
|--------|--------------|-------------------|
| **Duration** | 4 weeks | 10 weeks |
| **Focus** | Custom AI agents and workflows | Out-of-the-box Now Assist skills |
| **Customization** | High (built to customer requirements) | Low (configure, not build) |
| **Output** | Working prototype + roadmap | Production Now Assist deployment |
| **Complexity** | Higher (custom agents, integrations) | Lower (platform features) |
| **Customer maturity** | AI-ready, want differentiation | Getting started with ServiceNow AI |

### The Two-Track Approach

Some engagements combine both:
1. **Now Assist LAUNCH** — Get the basics running (summarization, search, code gen)
2. **FlightPath.AI** — Build custom agents on top of the foundation

---

## 6. The Delivery Model Big Picture

```
Customer Journey with newRocket AI:

ASSESS                ACTIVATE               SCALE
(FlightPath.AI)       (Implementation)        (Managed Services)

Week 1-4              Month 2-4               Ongoing
├── Discover          ├── Production deploy   ├── Monitor & optimize
├── Prototype         ├── User training       ├── Add new use cases
├── Demonstrate       ├── Change management   ├── Expand to new agents
└── Roadmap           └── Stabilization       └── Value reporting

     ↑                      ↑                       ↑
     FDE role:              FDE role:               FDE role:
     Lead technical         Build & deploy          Consult, optimize,
     discovery & sprint     production solution     feed back to product
```

### How FDE Work Feeds Product

This is a critical part of the role — the **product feedback loop**:

```
Client engagement:
  → You solve a problem for Client A with a custom integration
  → You notice Client B had a similar problem
  → You abstract the solution into a reusable component
  → You submit it to the Intelligence Platform team
  → It becomes a standard spoke/action/agent enhancement
  → Future FDEs deploy it in hours instead of days

This cycle is how newRocket builds defensible IP from services revenue.
```

### What Makes a Great FDE at newRocket

1. **Speed without shortcuts** — Ship fast, but production-grade. No demo-ware.
2. **Customer empathy** — Understand their business, not just their technology.
3. **Product instinct** — Spot patterns across engagements. Think "reusable, not one-off."
4. **Communication clarity** — Explain AI tradeoffs to non-technical stakeholders.
5. **Technical range** — ServiceNow scripting on Monday, RAG pipeline on Tuesday, Slack integration on Wednesday, executive demo on Thursday.
6. **Comfort with ambiguity** — Week 1 of every engagement is "figure out what to build." You don't get a spec.
