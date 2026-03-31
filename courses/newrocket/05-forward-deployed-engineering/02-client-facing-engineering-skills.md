# 02 – Client-Facing Engineering Skills

The technical skills get you to the table. The client-facing skills determine whether you succeed at this role. This module covers the non-coding competencies that separate great FDEs from good engineers.

---

## 1. Discovery Sessions

### What They Are

Discovery sessions are structured conversations where you uncover the customer's real problems, existing workflows, data landscape, and constraints. They happen in Week 1 of every FlightPath.AI engagement.

### Running a Discovery Session

**Before the session:**
```
Prepare:
├── Review customer's ServiceNow instance (if you have access)
├── Research their industry and common pain points
├── Understand which newRocket agents might apply
├── Prepare a lightweight agenda (not a slide deck)
└── Identify who's in the room and their roles
```

**During the session:**
```
Structure (60-90 minutes):
├── 5 min: Introductions and agenda
├── 20 min: Current state walkthrough
│   "Walk me through what happens when an employee reports an IT issue today."
│   "Where does it break down? Where do people get frustrated?"
│
├── 20 min: Data and systems landscape
│   "What's in your knowledge base? When was it last updated?"
│   "What monitoring tools feed into ServiceNow?"
│   "What systems would the AI need to connect to?"
│
├── 20 min: Success criteria and constraints
│   "If this works perfectly, what does that look like in 6 months?"
│   "What are the guardrails? What should AI never do?"
│   "What's been tried before? What didn't work?"
│
└── 10 min: Next steps and access requests
    "I'll need access to your dev instance and a sample of recent tickets."
```

**After the session:**
```
Deliverables:
├── Summary email within 24 hours
├── Identified pain points (prioritized)
├── Preliminary agent/solution mapping
├── Data quality concerns
├── List of access/resources needed
└── Proposed architecture sketch (rough)
```

### Discovery Anti-Patterns

| Anti-Pattern | Why It's Bad | What to Do Instead |
|-------------|-------------|-------------------|
| **Jumping to solutions** | You haven't understood the problem yet | Listen for 70% of the meeting, propose for 30% |
| **Leading the witness** | "You need Phoebe, right?" biases the conversation | "Tell me about your IT support challenges" |
| **Talking tech too early** | VP of IT doesn't care about GlideRecord | Start with business outcomes, drop to tech with technical stakeholders |
| **Ignoring the politics** | The project can fail even if the tech works | Ask "Who else should be involved? Who has concerns?" |
| **Over-promising** | "AI can fix everything" sets you up to fail | "Here's what's realistic in 4 weeks" |

---

## 2. Technical Workshops

### What They Are

Hands-on working sessions where you build alongside customer's technical team. These happen in weeks 2-3 of FlightPath.AI.

### Workshop Format

```
Technical Workshop (2-4 hours):

1. Architecture Review (30 min)
   ├── Present proposed solution architecture
   ├── Walk through data flow
   ├── Identify integration points
   └── Get feedback and alignment

2. Live Configuration (60-90 min)
   ├── Screen-share or pair-program in customer's instance
   ├── Configure agent topics and flows
   ├── Build integrations together
   └── Explain decisions as you make them
       ("I'm using an async Business Rule here because
        the API call takes 3-5 seconds and we don't
        want to block the form submission")

3. Testing and Iteration (30-60 min)
   ├── Test with real scenarios from customer's ticket history
   ├── Walk through edge cases together
   ├── Adjust configuration based on results
   └── Document what worked and what needs more work

4. Wrap-Up (15 min)
   ├── Summary of what was built
   ├── Open items and blockers
   ├── Plan for next session
   └── Action items (yours and theirs)
```

### Teaching While Building

A key FDE skill: you're not just building — you're **enabling the customer's team** to maintain and extend what you build. This means:

- Narrate your thought process as you code
- Explain ServiceNow patterns and why you chose them
- Point out where they can customize after you leave
- Create documentation as you go (not after)

---

## 3. Stakeholder Communication

### Know Your Audience

Every engagement has multiple stakeholders with different concerns:

| Stakeholder | Their Concern | What They Want to Hear |
|-------------|--------------|----------------------|
| **Executive Sponsor** (VP/CIO) | ROI, strategic value, risk | "We'll reduce support costs by $X and improve employee satisfaction by Y%" |
| **IT Director/Manager** | Impact on their team, adoption | "Your team will handle fewer L1 tickets and focus on complex issues" |
| **ServiceNow Admin** | Technical quality, maintainability | "The scoped app follows platform best practices and we'll document everything" |
| **End Users** | Will this actually help me? | "You'll get answers to common questions in seconds instead of waiting hours" |
| **Security/Compliance** | Data privacy, governance, audit | "All AI decisions are logged, PII is filtered, and we use AI Control Tower for governance" |
| **Finance** | Cost, budget, licensing | "Here's the token cost projection and the estimated FTE savings" |

### Communication Cadences

```
Daily:   Async update to project channel (Slack/Teams)
         "Today: Built REST integration for monitoring data.
          Blocker: Need API credentials for Datadog.
          Tomorrow: Start RAG pipeline configuration."

2-3 days: Live demo to technical team
          Show working feature, get feedback, adjust

Weekly:   Status update to sponsor
          Progress vs. plan, metrics, risks, decisions needed

End of Sprint: Executive readout
               Value delivered, metrics, roadmap recommendation
```

### Presenting Technical Tradeoffs

Non-technical stakeholders need to make technical decisions. Frame tradeoffs in business terms:

```
BAD:  "We can use a cross-encoder reranker which has O(n*q) complexity
       but better precision, or a bi-encoder with O(n+q) which scales better."

GOOD: "Option A gives 15% more accurate answers but costs 3x more to run
       and takes longer to respond. Option B is faster and cheaper but
       occasionally misses the best answer. For your volume of 500 tickets/day,
       Option B saves $200/month and the accuracy difference won't be
       noticeable to users. I recommend B for now, with the option to
       upgrade later if needed."
```

---

## 4. Rapid Prototyping Mindset

### The FDE Prototyping Philosophy

| Principle | What It Means |
|-----------|--------------|
| **Working software over perfect architecture** | Get something running in day 1 of the sprint |
| **Real data over mock data** | Use customer's actual tickets, articles, and data |
| **Visible value first** | Build the user-facing part before the infrastructure |
| **Iterate in hours, not weeks** | Show progress daily, adjust based on feedback |
| **Production path, not throwaway** | Prototype should evolve into production, not be rebuilt |

### The First Day Sprint

Your goal on the first day of building:

```
By end of Day 1 (Week 2):
├── Agent scoped app installed in customer dev instance
├── Basic topic configured (e.g., "check incident status")
├── At least one working conversation flow
├── Connected to customer's knowledge base (even if imperfect)
└── Can demo a simple end-to-end interaction

Why: This builds customer confidence and gives you something
     to iterate on instead of starting from zero on Day 2.
```

### Iteration Rhythm

```
Morning: Build the next feature/improvement
  ↓
Midday: Quick demo to technical stakeholder
  ↓
Afternoon: Adjust based on feedback, fix issues
  ↓
End of day: Async update, plan tomorrow
  ↓
Repeat
```

---

## 5. Managing Ambiguity

### Why Ambiguity is the Default

Week 1 of every engagement starts with ambiguity:
- Customer knows they "want AI" but not specifically what
- Requirements are vague or conflicting
- Data quality is unknown until you look at it
- Multiple stakeholders have different priorities
- Technical constraints emerge only during discovery

### Structuring the Undefined

**Step 1: Map the problem space**
```
"Your IT support challenge has several dimensions:
1. High ticket volume (you said 500/day)
2. Slow resolution (4.5 hour MTTR for P3)
3. Misrouting (30% of tickets go to the wrong group)
4. Stale knowledge base (last updated 18 months ago)
5. No self-service (users always open tickets)

Which of these hurts the most? That's where we start."
```

**Step 2: Propose a focused starting point**
```
"I recommend we start with self-service resolution for the top 5 ticket types.
This is: high impact (addresses #1 and #2), lower technical risk (knowledge-grounded,
not autonomous), and delivers visible value fast (users see results immediately)."
```

**Step 3: Define what "done" looks like for the sprint**
```
"In 4 weeks, success means:
- AI agent handles password resets, VPN issues, and email setup end-to-end
- 40% deflection rate on those 3 ticket types
- Customer satisfaction ≥ 4.0/5.0 on AI interactions
- Architecture documented for expanding to more ticket types"
```

**Step 4: Make the unknown known (as fast as possible)**
```
Day 1 tasks:
├── Audit knowledge base quality (is there enough to ground the AI?)
├── Analyze top 20 ticket types by volume (where's the impact?)
├── Test ServiceNow AI Search on sample queries (does retrieval work?)
└── Map integration requirements (what external systems are needed?)
```

---

## 6. Building Trust

### Trust is the FDE's Currency

In a 4-week sprint, you don't have time to build trust slowly. These patterns accelerate it:

**Be transparent about limitations:**
> "The AI won't be perfect on day one. Here's our plan for improving accuracy over the first month."

**Show, don't tell:**
> Don't promise results — demo them. A working prototype is worth 100 slides.

**Own problems early:**
> "I found an issue with your CMDB data quality that will affect the AI agent's accuracy. Here's what I recommend we do about it before go-live."

**Follow through on small things:**
> If you say "I'll send a summary by end of day," send it by end of day. Every time.

**Ask good questions:**
> Questions that show you understand their business build more trust than answers that show you know technology.

**Leave them better than you found them:**
> Even if the AI isn't perfect, if their knowledge base is better, their workflows are cleaner, and their team understands the technology — you've delivered value.

---

## 7. The Product Feedback Loop

### What to Report Back to newRocket

After every engagement, you should capture:

```
Product Feedback Report:
├── What worked well
│   ├── Which agents deployed (and how they performed)
│   ├── Which integrations were straightforward
│   └── What customer reactions were positive
│
├── What was difficult
│   ├── Configuration gaps (what should be easier)
│   ├── Missing features (what we had to build custom)
│   ├── Documentation gaps
│   └── Performance issues
│
├── Reusable patterns identified
│   ├── Custom spoke actions that should become standard
│   ├── Common workflow patterns across customers
│   ├── RAG configuration templates by industry
│   └── Guardrail rule sets by use case
│
└── Customer requests for roadmap
    ├── Features they asked for
    ├── Agents they want that don't exist yet
    └── Integrations they need
```

### Why This Matters

This feedback loop is what transforms newRocket from a services company into a product company. Every engagement should make the next engagement faster. Your pattern recognition across customers is one of the most valuable things you bring.
