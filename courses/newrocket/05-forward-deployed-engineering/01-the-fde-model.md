# 01 – The Forward Deployed Engineering Model

The FDE role is a distinct discipline — not a hybrid of SWE and consulting, but its own thing. Understanding the model deeply helps you position yourself in interviews and sets expectations for day one.

---

## 1. Origins and Evolution

### Palantir: Where It Started

Palantir created the FDE role (originally called "Delta") in the early 2010s to solve a specific problem: their software was powerful but required deep customization for each government/enterprise client. Traditional sales engineers couldn't build production systems. Traditional software engineers couldn't work directly with customers. FDEs did both.

**Key insight:** The most valuable engineer isn't always the one writing the core product — it's the one who makes the product work in the real world.

Until ~2016, Palantir had **more FDEs than traditional software engineers**. That's how central the role was to their business model.

### Who Else Uses This Model

| Company | FDE Focus |
|---------|-----------|
| **Palantir** | Data platform deployment for defense + enterprise |
| **OpenAI** | Three phases: scoping → validation → delivery |
| **Anthropic** | Enterprise AI deployment |
| **Ramp** | Financial infrastructure customization |
| **Salesforce** | AI agent deployment (Agentforce) |
| **Commure** | Healthcare software deployment |
| **Gecko Robotics** | Industrial robotics + software |
| **newRocket** | ServiceNow AI agent deployment |

### Why It's Exploding Now

FDE job postings up **800–1000% in 2025**. a16z called it "the hottest job in tech." The driver: AI products are general-purpose but require domain-specific deployment. Every company selling AI to enterprises needs people who can bridge the gap.

---

## 2. FDE vs. Everything Else

### The Spectrum

```
                    ← More Product            More Customer →

Software Engineer ──── Platform Engineer ──── FDE ──── Solutions Engineer ──── Consultant
     │                      │                  │              │                     │
  Builds core         Builds platform     Deploys &      Demos &              Advises
  product             features            customizes     configures           on strategy
     │                      │                  │              │                     │
  No customer         Indirect customer   Direct customer  Pre-sale            No building
  contact             exposure            embedded         focused             (usually)
```

### Detailed Comparison

| Dimension | SWE | FDE | Solutions Engineer | Management Consultant |
|-----------|-----|-----|-------------------|----------------------|
| **Where you work** | Company office/remote | Customer site/remote | Sales meetings | Client office |
| **Who you talk to** | Product team | Customer engineers + execs | Prospects | Customer execs |
| **What you build** | Product features | Customer solutions | Demos, POCs | Slide decks, strategies |
| **Code quality bar** | Production | Production | Demo-quality | N/A |
| **Problem definition** | Spec/story | Undefined (you define it) | Pre-defined (product fit) | Defined by engagement scope |
| **Success metric** | Feature shipped | Customer outcome | Deal closed | Report delivered |
| **Variety** | Deep in one codebase | New problem every month | Same product, different audiences | New engagement every few months |
| **Compensation range** | $150K–$400K+ | $120K–$340K+ | $100K–$250K | $130K–$300K+ |

### The FDE Superpower: Context Switching

In a single week, you might:
- Monday: Discovery call with a healthcare company's IT leadership
- Tuesday: Write GlideRecord scripts to customize incident routing
- Wednesday: Build a RAG pipeline connecting their Confluence to ServiceNow
- Thursday: Present architectural tradeoffs to the CTO
- Friday: Document reusable patterns and submit product feedback

This requires **technical breadth, communication range, and comfort with ambiguity** that most pure engineering or pure consulting roles don't demand.

---

## 3. The Client Engagement Lifecycle

### Phase 1: Scoping (Pre-Engagement)

Before you start building, you need to understand:

```
Technical Discovery:
├── What ServiceNow modules are they running?
├── How mature is their CMDB?
├── What's their knowledge base quality like?
├── What external systems need to integrate?
├── What's their data security posture?
└── Who are the technical stakeholders?

Business Discovery:
├── What pain points are driving this initiative?
├── What does success look like to the sponsor?
├── What are the political dynamics? (Who's for/against this?)
├── What's the timeline pressure?
└── What have they tried before? (Why did it fail?)
```

### Phase 2: Building (The Sprint)

This is where you spend most of your time. Within newRocket's FlightPath.AI, this is weeks 2-3:

```
Daily Rhythm:
├── Morning standup with customer team (15 min)
├── Heads-down building (4-5 hours)
├── Afternoon demo/check-in with stakeholder (30-60 min)
├── Internal sync with newRocket team (15 min)
└── Documentation and planning (30-60 min)
```

### Phase 3: Delivery and Handoff

```
Deliverables:
├── Working system in customer's ServiceNow instance
├── Architecture documentation
├── Runbook for operations team
├── Training materials for customer admins
├── Value metrics dashboard
├── Product feedback report for newRocket
└── Recommendations for phase 2
```

### Phase 4: Transition

You move to the next engagement. The customer either:
- Goes to newRocket's managed services team for ongoing support
- Has their internal team take over (with your documentation)
- Engages for a phase 2 sprint for the next set of use cases

---

## 4. The Embedded vs. Rotational Models

### Embedded FDE
- **Duration:** 3-12 months with one customer
- **Depth:** Very deep understanding of customer's environment
- **Risk:** Customer dependency, slower product feedback cycle
- **Used by:** Palantir (traditionally), Commure

### Rotational FDE (newRocket's Model)
- **Duration:** 4-week sprints per customer (FlightPath.AI)
- **Depth:** Focused but time-boxed
- **Benefit:** More diverse experience, faster pattern recognition across customers
- **Risk:** Less time to go deep, steeper ramp per engagement
- **Used by:** newRocket, OpenAI (modified)

### What This Means for You

You'll likely work with **6-12 different customers per year**, each for 4-8 weeks. This means:
- You need to ramp fast on new environments
- Your ServiceNow fundamentals need to be solid (no time to relearn)
- Pattern recognition accelerates over time (customer 5 is much easier than customer 1)
- You build a broad network across industries and company sizes

---

## 5. Skills That Differentiate Great FDEs

### Technical Skills (Table Stakes)

These get you in the door:
- Full-stack development (JS/TS, Python, APIs, databases)
- AI/LLM integration (RAG, agents, embeddings, prompt engineering)
- Cloud platform proficiency (at least one of AWS/Azure/GCP)
- ServiceNow basics (can be learned on the job, but helps)

### Differentiating Skills (What Gets You Hired and Promoted)

**1. Problem Decomposition Under Ambiguity**

The customer says: "We want AI to improve our IT support."

You need to turn that into:
```
Specific problems:
├── Ticket categorization is slow and inconsistent (30% misrouted)
├── Knowledge base is 18 months stale
├── P3/P4 tickets consume 60% of agent time
├── No self-service resolution for common issues (password, VPN, email)
└── MTTR is 4.5 hours for P3 tickets

Prioritized solutions (by impact × feasibility):
1. AI-powered self-service for top 10 ticket types (quick win, high impact)
2. Auto-categorization + routing (reduces misrouting, measurable)
3. Knowledge article generation from resolved tickets (improves data quality)
4. Predictive routing based on skills and workload (moderate complexity)
```

**2. Technical Communication Across Levels**

Same topic, three audiences:

| Audience | How You Say It |
|----------|---------------|
| **VP of IT** | "We'll reduce your mean time to resolution by 40% and free up 2 FTEs worth of agent capacity in 90 days." |
| **IT Manager** | "The AI agent handles L1 tickets — password resets, VPN, email setup. Your team focuses on complex issues. We'll start with the top 5 ticket types and expand." |
| **ServiceNow Admin** | "We're deploying a scoped app with a Virtual Agent topic, RAG pipeline to your KB, and Flow Designer integration to auto-create categorized incidents. I'll need admin access to your dev instance." |

**3. Rapid Prototyping**

Your goal in week 2-3 of a FlightPath.AI sprint: **a working demo with real data, not slides**. This means:
- Speed over perfection (iterate, don't architect)
- Real data, not mock data (builds customer confidence)
- Visible value, not invisible infrastructure (show the end-user experience first)
- Working in the customer's actual ServiceNow instance (not your sandbox)

**4. Product Judgment**

Every engagement produces custom work. The question is: **what part of this should become product?**

```
One-off: Custom integration with customer's proprietary ERP system
Reusable: A pattern for connecting any ERP to ServiceNow via REST + CMDB mapping

One-off: Specific escalation rules for this customer's org chart
Reusable: A configurable escalation framework that any customer can parameterize

One-off: Ticket routing to this customer's 47 assignment groups
Reusable: An ML-based routing model that trains on any customer's historical data
```

**5. Managing Customer Relationships**

- **Set expectations early.** "In 4 weeks, we'll have a working prototype, not a production system."
- **Demo frequently.** Show progress every 2-3 days, not just at the end.
- **Own bad news.** "The knowledge base quality is blocking us. Here's what we need to fix before AI will work well."
- **Leave them better than you found them.** Documentation, training, and a clear path forward.
