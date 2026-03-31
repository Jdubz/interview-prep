# 02 – System Design and Scenarios

Practice scenarios that simulate the kinds of challenges you'll face as an FDE. Work through these like mini-interviews — structure your thinking, make decisions, and articulate tradeoffs.

---

## 1. System Design: AI-Powered IT Support

### The Prompt

> "Design an AI-powered IT support system for a 10,000-employee company using ServiceNow. They currently handle 500 tickets/day with 30% misrouting and a 4.5-hour average resolution time for P3 tickets. They want to reduce resolution time by 50% and achieve 40% ticket deflection."

### Structured Response

**Step 1: Clarify Requirements (2-3 minutes)**

Ask before designing:
- What ServiceNow modules are already in place? (ITSM, CMDB, Knowledge Management)
- How mature is their knowledge base? (If poor, that's a constraint)
- What external systems integrate with ServiceNow? (Monitoring, chat, email)
- What's their tolerance for AI autonomy? (Suggest only? Auto-resolve?)
- What compliance requirements exist? (Healthcare? Finance? Government?)

**Step 2: Architecture (5-7 minutes)**

```
┌──────────────────────────────────────────────────────────┐
│                    Entry Points                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ Chat/VA  │  │  Portal  │  │  Email   │  │  Phone   ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│       └──────────────┴──────────────┴──────────────┘      │
│                          │                                │
│                ┌─────────▼──────────┐                     │
│                │  AI Triage Agent   │                     │
│                │  (Phoebe variant)  │                     │
│                │  ├── Classify      │                     │
│                │  ├── Route         │                     │
│                │  └── Self-serve?   │                     │
│                └─────────┬──────────┘                     │
│                          │                                │
│              ┌───────────┼───────────┐                    │
│              │           │           │                    │
│    ┌─────────▼───┐ ┌────▼────┐ ┌───▼──────────┐        │
│    │ Self-Service│ │  Assist │ │   Escalate   │        │
│    │ Resolution  │ │  Agent  │ │   to Human   │        │
│    │ (auto)      │ │ (suggest)│ │ (with context)│        │
│    └─────────────┘ └─────────┘ └──────────────┘        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Supporting Services                                │  │
│  │  ├── RAG Pipeline (Knowledge Base + past incidents) │  │
│  │  ├── CMDB Enrichment (affected CI, dependencies)    │  │
│  │  ├── Predictive Intelligence (classification model) │  │
│  │  └── AI Control Tower (governance, audit, metrics)  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Value Dashboard                                    │  │
│  │  ├── Deflection rate, MTTR, CSAT                   │  │
│  │  ├── Token costs, escalation rate                   │  │
│  │  └── Knowledge gap analysis                         │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Step 3: Data Flow for a Typical Interaction (3-4 minutes)**

```
User: "I can't connect to VPN from my laptop"

1. INTAKE
   ├── User types in chat (Virtual Agent on Service Portal)
   ├── AI Triage Agent receives message
   └── Classify intent: "vpn_connectivity" (confidence: 0.91)

2. ENRICHMENT
   ├── Look up user in sys_user → get department, location, role
   ├── Look up user's laptop in CMDB → model, age, OS, last patch
   ├── Check active incidents → any VPN outages? (Event Management)
   └── Check change calendar → recent changes to VPN infrastructure?

3. CONTEXT-AWARE ROUTING
   ├── Known outage? → "We're aware of a VPN issue. ETA for fix: 2 hours."
   ├── No outage → Search knowledge base for VPN troubleshooting
   └── RAG returns: KB0045678 (score: 0.87) — "VPN Connection Troubleshooting"

4. SELF-SERVICE ATTEMPT
   ├── Present guided troubleshooting steps from KB article
   ├── "Have you tried: 1) Restart VPN client 2) Check internet connection
   │    3) Clear VPN cache (instructions attached)"
   └── Ask: "Did this resolve your issue?"

5a. RESOLVED (deflection — 40% of cases)
    ├── Log interaction in x_myco_ai_interaction
    ├── Increment deflection counter
    ├── Ask for satisfaction rating
    └── Update knowledge article confidence score

5b. NOT RESOLVED (escalation — 60% of cases)
    ├── Create incident (auto-categorized: Network > VPN > Connectivity)
    ├── Attach: user info, CMDB data, conversation history, KB articles tried
    ├── Route to "Network Support" assignment group
    ├── AI suggests resolution to human agent based on similar past incidents
    └── SLA clock starts from incident creation (not from chat start)
```

**Step 4: Address the Requirements (2-3 minutes)**

| Requirement | How We Hit It |
|-------------|-------------|
| **50% MTTR reduction** (4.5h → 2.25h) | Self-service resolves simple cases instantly. For escalated cases, AI pre-categorizes and enriches, saving 30-60 min of agent triage time. AI suggests resolutions from similar past incidents. |
| **40% ticket deflection** | Self-service resolution for top ticket types (password reset, VPN, email setup). Knowledge-grounded answers deflect users before they create tickets. |
| **30% misrouting → <10%** | ML classification model trained on 100K+ historical incidents. Auto-categorization with 90%+ accuracy. Route to assignment group based on category, not user guesswork. |

**Step 5: Non-Functional Concerns (2 minutes)**

- **Security:** ACL enforcement in RAG (users only see knowledge they're authorized for). PII filtering in AI responses. Audit trail for all AI decisions.
- **Scale:** 500 tickets/day ≈ 60/hour. Async AI pipeline handles easily. Token budget: ~$15/day at 1,500 tokens/interaction.
- **Reliability:** Fallback to human routing if AI service is down. Circuit breaker on LLM API calls. Graceful degradation.
- **Rollout:** Pilot with IT department first, measure for 2 weeks, expand gradually.

---

## 2. Scenario: Client Discovery Gone Wrong

### The Prompt

> "You arrive at a new client for a FlightPath.AI engagement. In the discovery session, you learn: their knowledge base hasn't been updated in 2 years, their CMDB was last audited 18 months ago, and the IT director is skeptical about AI ('we tried chatbots before and they were terrible'). How do you handle this?"

### Response Framework

**Acknowledge the reality (don't dismiss concerns):**
> "Those are legitimate concerns. Previous chatbot failures usually happened because the technology wasn't ready, or the implementation tried to boil the ocean. We're going to take a different approach."

**Address the data problem honestly:**
> "The knowledge base and CMDB quality are genuine blockers. I won't pretend AI will work well on top of stale data — it won't. Here's what I recommend:
>
> Week 1: We do a data quality assessment. I'll analyze your top 20 ticket types by volume, check which ones have matching knowledge articles, and score the articles for accuracy and completeness. I'll also audit your CMDB for the CIs most relevant to your top incident categories.
>
> Week 2-3: We tackle this in parallel:
> - Fix the top 5 knowledge gaps (write/update the articles that cover 40% of your ticket volume)
> - Set up automated article generation from resolved incidents (so the knowledge base improves continuously)
> - Deploy AI on the topics where knowledge IS solid (there are always some)
>
> Week 4: Demo what's working, show the data quality improvement trajectory, and present a 90-day plan to get to target coverage."

**Address the skepticism directly:**
> "You mentioned chatbots were terrible before. Can you tell me more about what happened? [Listen]. The difference with this approach: we're not building a general-purpose chatbot that makes things up. We're building a knowledge-grounded system that only answers questions it has solid source material for. When it doesn't know, it escalates to your team with full context — it doesn't hallucinate. And I'll prove this in week 2 with real tickets from your environment."

**Set realistic expectations:**
> "In 4 weeks, we won't transform your entire IT support operation. But we will: have AI handling 3-5 specific ticket types end-to-end, demonstrate measurable deflection, and give you a clear roadmap with data quality milestones. The ROI case will be built on real numbers from your environment, not vendor slides."

---

## 3. Scenario: Technical Architecture Decision

### The Prompt

> "A customer wants their AI agent to have access to data in Confluence, SharePoint, and their internal wiki (custom Django app) in addition to ServiceNow's knowledge base. How would you architect this?"

### Response

```
Architecture: Multi-Source RAG Pipeline

┌─────────────────────────────────────────────────┐
│              Data Sources                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────────┐ │
│  │ ServiceNow│ │Confluence │ │  SharePoint   │ │
│  │    KB     │ │  (Cloud)  │ │  (O365)       │ │
│  └─────┬─────┘ └─────┬─────┘ └──────┬────────┘ │
│        │              │              │           │
│  ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴────────┐ │
│  │ Native    │ │ REST API  │ │ Graph API    │ │
│  │ AI Search │ │ Connector │ │ Connector    │ │
│  └─────┬─────┘ └─────┬─────┘ └─────┬────────┘ │
│        │              │              │           │
│  ┌─────┴──────────────┴──────────────┴────────┐ │
│  │        Unified Embedding Pipeline           │ │
│  │  ├── Normalize content format               │ │
│  │  ├── Chunk (source-aware strategy)          │ │
│  │  ├── Embed (same model for all sources)     │ │
│  │  ├── Tag metadata (source, date, access)    │ │
│  │  └── Store in vector database               │ │
│  └─────────────────────┬──────────────────────┘ │
│                        │                         │
│  ┌─────────────────────▼──────────────────────┐ │
│  │        Unified Retrieval                    │ │
│  │  ├── Query all sources simultaneously       │ │
│  │  ├── Score and rerank across sources         │ │
│  │  ├── Apply access control filters           │ │
│  │  └── Deduplicate (same content in multiple  │ │
│  │       sources)                               │ │
│  └─────────────────────┬──────────────────────┘ │
│                        │                         │
│  ┌─────────────────────▼──────────────────────┐ │
│  │        Generation                           │ │
│  │  ├── Top chunks with source citations       │ │
│  │  ├── "According to [Confluence: Page Title]" │ │
│  │  └── Multi-source synthesis                 │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Django Wiki (Custom):**
> "For the custom Django wiki, I'd build a lightweight API endpoint on their side that returns articles in a standard format (title, body, metadata, last_updated). Then I'd write a custom Integration Hub spoke that calls this API on a schedule, pulls content, and feeds it into the same embedding pipeline. If they can't modify the Django app, we can scrape it or read directly from their database via MID Server + JDBC."

**Key tradeoffs to discuss:**
- **Freshness:** Confluence and SharePoint have webhooks — we can re-embed content on change. The Django wiki needs polling (scheduled sync).
- **Access control:** Each source has its own permission model. We need to tag embeddings with access metadata and filter at retrieval time. This is the hardest part.
- **Deduplication:** The same troubleshooting guide might exist in ServiceNow KB AND Confluence. The retrieval layer needs dedup logic.
- **Latency vs. coverage:** Searching 4 sources is slower than 1. We can parallelize, but the reranking step needs to handle heterogeneous quality.

---

## 4. Scenario: Stakeholder Conflict

### The Prompt

> "The IT Director wants the AI agent to auto-resolve P3 and P4 tickets without human review. The Security team says no autonomous AI actions — everything needs human approval. How do you navigate this?"

### Response

> "Both sides have valid concerns. The IT Director wants efficiency — P3/P4 tickets are drowning the team. Security wants control — autonomous AI actions introduce risk.
>
> My proposal: **graduated autonomy based on confidence and category.**
>
> **Tier 1 — Full auto-resolve (no human review):**
> - Confidence > 95%
> - Category is well-known (password reset, VPN reconnect, email setup)
> - Action is low-risk (provide information, send link, trigger standard workflow)
> - Builds trust over time as we prove accuracy
>
> **Tier 2 — AI suggests, human one-click approves:**
> - Confidence 80-95%
> - Category is moderate complexity
> - Agent sees the AI suggestion and approves with one click (not re-do the work)
> - This is the 'training wheels' mode
>
> **Tier 3 — AI enriches, human resolves:**
> - Confidence < 80%
> - Complex or sensitive categories
> - AI provides context (CMDB data, similar incidents, knowledge articles) but doesn't suggest resolution
>
> The security team gets: audit trails, governance rules, human-in-the-loop for anything uncertain, and a kill switch.
>
> The IT Director gets: immediate auto-resolution for the simple stuff (40% of volume), faster resolution for the rest (agent does less triage work).
>
> We start with Tier 2 for everything for the first 2 weeks, measure accuracy, and then unlock Tier 1 for high-confidence categories based on real data. That way we're making a data-driven decision, not a theoretical one."

---

## 5. Portfolio Preparation

### What to Have Ready

You should be able to walk through 2-3 projects in detail. For each, prepare:

```
Project Template:
├── Context: Company, team size, timeline, your role
├── Problem: What business problem were you solving?
├── Technical approach: Architecture, tech stack, key decisions
├── Challenges: What went wrong or was unexpectedly hard?
├── Results: Concrete metrics (latency, cost, adoption, etc.)
├── What you'd do differently: Shows growth and self-awareness
└── How it applies to this role: Connect to FDE/newRocket specifically
```

### Mapping Your Projects to newRocket's Needs

| newRocket Need | Your Project Should Show |
|---------------|------------------------|
| AI/LLM integration | You've integrated LLMs into production systems |
| Client-facing work | You've worked directly with customers/stakeholders |
| Enterprise systems | You've built integrations, dealt with auth/security |
| Rapid prototyping | You've shipped something fast under pressure |
| Product contribution | You've built reusable components, not just one-offs |
| Technical communication | You've explained complex systems to non-technical people |

---

## 6. Interview Day Checklist

```
Before:
□ Read this course (especially Modules 00 and 04)
□ Have 3 project stories mapped to STAR format
□ Prepare 5+ questions for each interviewer
□ Know newRocket's recent news (last 3 months)
□ Know the names: Harsha Kumar (CEO), Frank Palermo (COO)
□ Have a clear answer for "why newRocket" and "why FDE"
□ Test your AV setup (camera, mic, lighting, background)

During:
□ Take a breath before answering (it's OK to pause)
□ Structure long answers: "There are 3 parts to this..."
□ Draw diagrams if you can (screen share, virtual whiteboard)
□ Ask clarifying questions before diving into system design
□ When you don't know something, say so and explain how you'd learn it
□ Connect your answers back to their role/products when natural

After:
□ Send thank-you emails within 24 hours
□ Reference something specific from each conversation
□ If you realized you could have answered something better, mention it
□ Follow up on any commitments you made ("I said I'd send that article...")
```

---

## 7. The Anxiety Plan

> **Context:** High-pressure interviews can trigger freeze responses. Having a plan makes the freeze less likely and gives you a recovery path if it happens.

### Pre-Interview Routine
- 30 minutes before: review your 3 project stories (not cramming, just warming up)
- 10 minutes before: close all other apps, get water, take 5 deep breaths
- 2 minutes before: remind yourself of one thing you're genuinely excited about in this role

### If You Freeze Mid-Question
1. **Say it out loud:** "Let me take a moment to organize my thoughts."
2. **Buy time with structure:** "There are a few angles to this. Let me start with..."
3. **Start with what you know:** Even a partial answer is better than silence. Begin with what's clear and build from there.
4. **Ask for clarification:** "Could you help me understand which part you'd like me to focus on?" This is a legitimate FDE skill — clarifying before building.

### If You Don't Know the Answer
> "I haven't worked with [specific technology] directly, but here's how I'd approach it based on [related experience]. In the role, I'd [specific plan to learn it]."

This is exactly what a good FDE does — they don't know everything about every customer's environment on day 1. The skill is knowing how to ramp.
