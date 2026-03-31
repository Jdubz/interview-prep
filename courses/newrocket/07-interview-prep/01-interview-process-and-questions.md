# 01 – Interview Process and Questions

What to expect from newRocket's interview process and the questions you'll need to nail. Organized by category with model answers.

---

## 1. newRocket's Interview Process

### What Glassdoor Reveals (14 reviews, 2.5/5 difficulty)

**Typical stages:**
```
Stage 1: Recruiter Screen (30 min)
├── Role fit, salary expectations, timeline
├── Basic background questions
└── Can happen within 2 days of application

Stage 2: Hiring Manager Interview (45-60 min)
├── Technical depth + behavioral questions
├── For this role: AI Center of Excellence Lead
├── Expect questions about AI experience, client work, ServiceNow curiosity
└── Your chance to ask about the team and roadmap

Stage 3: Team Interview (45-60 min)
├── Cross-functional — may include sister team members
├── Technical scenarios, collaboration style
└── How you'd work with existing team members

Stage 4: VP/Leadership Interview (30-45 min)
├── Strategic thinking, culture fit
├── Why newRocket, why this role, career goals
└── May discuss compensation details

Total timeline: 1-2 weeks typical (some report up to 3 weeks)
```

### What to Prepare

- **Resume deep-dive:** Be ready to go deep on any project listed
- **AI portfolio:** Concrete examples of AI/LLM systems you've built
- **Client stories:** Examples of working directly with customers on technical projects
- **ServiceNow awareness:** You don't need to be an expert, but showing you understand the platform is a differentiator
- **Company knowledge:** Understanding of newRocket's products, market position, and strategy

---

## 2. The 2-Minute Pitch

### Structure (15s / 60s / 25s)

> I'm Josh — I'm a senior full-stack engineer based in Portland with about 12 years of experience, and the through-line of my career has been building technical solutions in direct partnership with clients and stakeholders.
>
> I got into tech through an unusual path — I studied music at UC Santa Cruz and discovered programming through building interactive installations. I spent a few years as a technical director at Britelite Immersive, building live broadcast software for companies like Facebook, Instagram, and Salesforce. That work had zero tolerance for failure — millions of viewers, no second chances — and it taught me to build reliable systems under real pressure.
>
> From there I co-founded a consulting shop called Opna Development, where I was the face of the company for about five years. I scoped projects, pitched clients, designed architectures, and led delivery across a wide range of work — everything from enterprise Salesforce integrations with HIPAA compliance, to building NLP-powered chatbot systems on Dialogflow for McDonald's and JLL. That experience gave me the combination of client-facing skills and hands-on engineering that I think is pretty rare.
>
> Most recently I spent three years at Fulfil Solutions, a grocery robotics startup. I joined as employee 98 and helped scale the platform to support Amazon Fresh, DoorDash, and Uber Eats. I built a unified ordering API that handled integration with all of those partners through a single system — each with their own requirements, protocols, and failure modes. I also led the integration with Amazon directly, working closely with their team to translate their business requirements into our architecture. That project is now powering robotic fulfillment in Whole Foods stores across the US.
>
> What draws me to newRocket is the forward deployed AI engineer role specifically. It's the intersection of everything I've done — client-facing engineering, building integrations across complex enterprise systems, and now applying that to AI. I've been deep in the LLM and agentic AI space on my own, and the chance to deploy that at enterprise scale on a platform like ServiceNow, with a team that's purpose-built for it, is exactly where I want to be.

### Superpower Statement

"I build technical solutions in direct partnership with clients — scoping, architecting, delivering, and handing off. I've been doing forward deployed engineering before it had a name."

### Key Narrative Decisions

- **Dialogflow work is gold** — conversational AI for enterprise clients maps directly to what newRocket builds. Mention early if they ask follow-ups.
- **Opna chapter IS the FDE pitch** — scoping, client-pitching, architecting, delivering. Frame it that way.
- **Fulfil's Amazon integration** — enterprise integration under strict requirements, building reusable patterns across multiple partners (DoorDash, Uber Eats, Amazon). This is exactly what newRocket wants FDEs to do: ship for one client, generalize into platform.
- **Skip Meow Wolf and Madrone** — interesting but dilute the narrative for this role.
- **Skip CNC machinist background** — unless they ask about unusual paths, not relevant here.
- **Music → tech origin** — keep it to one sentence. It's a memorable hook, not the story.

### Adapting the Pitch Per Interviewer

| Interviewer | Adjust Ending To |
|-------------|-----------------|
| **Recruiter** | Keep it broad. Emphasize excitement about the role and company. |
| **AI CoE Lead (hiring manager)** | Lean into Dialogflow/NLP experience and your self-driven LLM learning. Ask about the agent architecture. |
| **Sister team / cross-functional** | Emphasize Opna's collaborative model and Fulfil's cross-team work (ops, marketing, Amazon's team). |
| **VP / leadership** | Lead with Opna co-founder experience and the Amazon integration outcome. Show strategic thinking. |

---

## 3. Top 25 Questions — Quick Reference

| # | Category | Question | Key Points |
|---|----------|----------|------------|
| 1 | Company | Why newRocket? | Pure-play ServiceNow, AI-first pivot, FDE model, speed-to-value |
| 2 | Role | Why FDE over traditional SWE? | Impact visibility, problem variety, technical + business skills |
| 3 | Role | How do you handle ambiguity? | Structure the undefined, propose focused starting points |
| 4 | Technical | Describe a RAG system you've built | End-to-end: data → chunking → embedding → retrieval → generation |
| 5 | Technical | How would you integrate an LLM into an enterprise system? | Auth, rate limiting, error handling, async, cost management |
| 6 | Technical | What's the difference between fine-tuning and RAG? | When to use each, cost/latency/accuracy tradeoffs |
| 7 | ServiceNow | What do you know about ServiceNow? | Platform architecture, tables/records, GlideRecord, Flow Designer |
| 8 | ServiceNow | How would you learn ServiceNow quickly? | PDI, newRocket University, learning by doing on engagements |
| 9 | AI | How do you evaluate AI system quality? | Relevancy, accuracy, latency, cost, user satisfaction, deflection rate |
| 10 | AI | How do you handle AI hallucinations? | RAG grounding, temperature, confidence thresholds, human-in-the-loop |
| 11 | AI | Explain agent architectures | Single agent, multi-agent, orchestration, A2A protocol |
| 12 | Client | Walk through a client engagement | Discovery → Architecture → Build → Demo → Handoff |
| 13 | Client | How do you manage stakeholders? | Know your audience, communicate at their level, set expectations |
| 14 | Client | A client disagrees with your technical recommendation. What do you do? | Listen, understand their concern, reframe as tradeoffs, find common ground |
| 15 | Behavioral | Tell me about a project that failed | What happened, what you learned, what you'd do differently |
| 16 | Behavioral | How do you prioritize competing demands? | Impact × feasibility matrix, communicate tradeoffs, get alignment |
| 17 | Behavioral | Describe a time you had to learn a new technology quickly | Specific example, your learning process, how you delivered |
| 18 | Technical | Design an AI-powered IT support system | Architecture, data flow, RAG pipeline, agent design, monitoring |
| 19 | Technical | How do you ensure AI systems are secure? | Input validation, ACLs, PII filtering, audit trails, governance |
| 20 | Technical | Explain vector embeddings to a non-technical person | Analogy: every document gets a GPS coordinate in meaning-space |
| 21 | Product | How do you decide what should be reusable vs. one-off? | Frequency across clients, abstraction cost, maintenance burden |
| 22 | Product | How would you improve newRocket's AI agent catalog? | Specific suggestion based on your understanding of the products |
| 23 | Integration | How would you connect ServiceNow to a customer's monitoring tool? | Integration Hub spoke, REST API, MID Server for on-prem, event-driven |
| 24 | Culture | What are your values? | Map to newRocket's: Excellence, Creativity, Integrity, Teamwork, Empathy |
| 25 | Closing | What questions do you have for us? | Prepared, thoughtful questions about team, roadmap, AI strategy |

---

## 3. Company & Role Questions (Detailed)

### Q1: "Why newRocket?"

**Framework:** Show you understand their positioning and it aligns with what you want.

> "Three things drew me to newRocket specifically. First, the pure-play ServiceNow focus — you're not spread across 10 platforms, which means deeper expertise and more interesting technical challenges. Second, the AI-first pivot is happening now, not in 3 years. The Intelligent Agent Crew, FlightPath.AI, the new leadership team — this is the inflection point, and I want to be part of building that. Third, the forward deployed model. I've seen how powerful it is when engineers are embedded with customers instead of throwing solutions over the wall. The combination of building real AI systems AND seeing them impact real businesses is exactly what I want."

### Q2: "Why forward deployed instead of a traditional engineering role?"

> "I've learned that I do my best work when I can see the impact directly. In traditional SWE, you ship a feature and maybe hear about adoption metrics months later. As an FDE, you build something in week 2 and see it solve a real problem in week 3. I also enjoy the variety — every engagement is a new problem space, new industry, new constraints. And frankly, the combination of technical depth and client-facing skills is rare and valuable. I'd rather build that combination than specialize narrowly."

### Q3: "What do you know about our products?"

> "I've researched your AI product line. The Intelligent Agent Crew has 9 purpose-built agents — Phoebe for IT support, Ariel for HR, Elara for knowledge management, Miles for IT ops, Heidi for finance, with security, managed services, platform, and telecom coming. What's interesting architecturally is the microservices-inspired design — a core agent with specialized helpers, and agent-to-agent collaboration via purpose-built protocols. FlightPath.AI is the delivery vehicle — a 4-week sprint from discovery to working prototype. And the Value Realization Dashboard ties everything to measurable business outcomes, which is how you prove ROI. I'm curious about how the product feedback loop from FDE engagements actually works in practice — how quickly do patterns from the field turn into platform features?"

---

## 4. Technical Questions (Detailed)

### Q4: "Walk me through a RAG system you've built or would build"

**Framework:** End-to-end architecture with ServiceNow context.

> "I'll frame this in the ServiceNow context since that's where I'd be building. The ingestion pipeline starts with ServiceNow's knowledge bases — articles get chunked into ~750-word segments with section-boundary awareness and overlap between chunks. Each chunk is embedded using ServiceNow's embedding model and stored in their vector database with metadata: source article, last updated date, category, and access control tags.
>
> At query time, the user's question goes through hybrid retrieval — both vector similarity search and BM25 keyword search. The results are fused and passed through a reranker that combines both scores. Top 3-5 chunks become the context for the LLM.
>
> The generation step uses a system prompt that constrains the model to answer only from provided context, with citations back to source articles. Confidence scoring determines whether to present the answer directly, suggest it to a human agent, or escalate.
>
> The critical non-obvious parts: ACL enforcement in retrieval so users only see articles they're authorized for, metadata filtering to exclude stale content, and a feedback loop where low-confidence or thumbs-down responses flag knowledge gaps."

### Q5: "How would you handle a customer whose knowledge base is outdated?"

> "This is actually one of the most common blockers in enterprise AI deployments. I'd approach it in phases.
>
> First, quantify the problem. Run an analysis of knowledge articles — when were they last updated, what percentage map to active incident categories, what's the gap between ticket volume by topic and knowledge coverage.
>
> Second, prioritize by impact. If the top 10 ticket types represent 60% of volume, focus on getting those 10 topics well-documented first. Don't try to fix everything at once.
>
> Third, automate knowledge creation. Set up a pipeline where resolved incidents with good resolution notes automatically generate draft knowledge articles. Elara, newRocket's knowledge agent, can help here. The drafts go to subject matter experts for review, not publication — you're not replacing human judgment, you're reducing the blank-page problem.
>
> Fourth, be transparent with the customer. 'The AI agent's accuracy will be limited until we improve the knowledge base. Here's our plan to get from 40% coverage to 85% over 6 weeks, and here's what performance will look like at each stage.'
>
> I'd set up quality metrics in the Value Realization Dashboard so the customer can see knowledge health improving over time."

### Q6: "Explain the difference between fine-tuning and RAG. When would you use each?"

> "They solve different problems. RAG is about giving the model access to information it doesn't have — your company's specific procedures, your CMDB data, your recent incidents. The model's weights don't change; you're just providing better context.
>
> Fine-tuning is about changing how the model behaves — its style, format, or domain-specific reasoning patterns. You're adjusting the model's weights with new training data.
>
> In the ServiceNow context: RAG for almost everything. When a customer asks 'How do I reset my VPN password?', the answer is in their knowledge base, not in the model's training data. RAG retrieves the right article and generates an answer grounded in it.
>
> Fine-tuning would make sense if you need the model to consistently follow a specific response format, use company-specific terminology correctly, or handle a classification task that requires deep domain understanding. But ServiceNow's Now LLM is already fine-tuned on ITSM terminology, so for most customers, RAG on top of Now LLM covers it.
>
> The practical tradeoffs: RAG is faster to set up, cheaper to maintain, and doesn't require training infrastructure. Fine-tuning gives better consistency but needs training data, compute, and ongoing maintenance as the domain evolves."

---

## 5. Behavioral Questions (Detailed)

### Q15: "Tell me about a project that failed or a major challenge you overcame"

**Framework:** Use STAR, but emphasize what you learned and how it applies to the FDE role.

> *Pick a real example. Structure: Situation → what went wrong → what you did → what you learned → how it applies.*
>
> Key elements to hit:
> - **Own the failure** — don't blame others
> - **Show the diagnosis** — you understood why it failed, not just that it failed
> - **Show adaptation** — what you changed in the moment
> - **Show learning** — how it changed your approach going forward
> - **Connect to FDE** — "This taught me to X, which is directly relevant to working with enterprise clients"

### Q16: "How do you handle a situation where a client's technical team pushes back on your recommendation?"

> "First, I listen. Their pushback usually comes from context I don't have — they've been burned by something similar before, there's a political dynamic, or there's a technical constraint I haven't discovered yet.
>
> Then I reframe. Instead of 'my way vs. their way,' I present it as tradeoffs: 'Here are the pros and cons of both approaches. My recommendation is X because of Y, but I hear your concern about Z. What if we tried a middle ground where we...?'
>
> If it's a genuine technical disagreement, I propose a quick test. 'Let's spend 2 hours building a small proof of concept for both approaches and let the data decide.' Engineers respect evidence more than authority.
>
> And sometimes they're right. I've changed my recommendation based on client feedback more than once. Being right is less important than shipping something that works in their environment."

### Q17: "Describe a time you had to learn something new under time pressure"

> *Pick a concrete example where you learned a new technology, framework, or domain quickly and delivered results. Emphasize:*
> - **How you learned** — documentation, prototyping, asking experts, reading source code
> - **How fast** — give a specific timeline
> - **What you shipped** — concrete outcome, not just "I learned it"
> - **What was hard** — shows self-awareness
>
> *Then connect: "This is basically every FDE engagement — Week 1 is always 'learn the customer's environment and figure out what to build.' I'm comfortable with that ramp because I've practiced it."*

---

## 6. Questions to Ask Them

### Technical Questions
- "How is the AI Center of Excellence structured? How many FDEs, and how do you share knowledge across engagements?"
- "What's the typical tech stack for an FDE engagement? How much is ServiceNow-native vs. custom code?"
- "How does the product feedback loop work in practice? When an FDE identifies a reusable pattern, what's the path to it becoming a platform feature?"
- "What's the biggest technical challenge the Agentic AI team is facing right now?"

### Role-Specific Questions
- "What does a successful first 90 days look like for this role?"
- "How many FlightPath.AI engagements would I be on simultaneously?"
- "What's the ratio of client-facing time vs. heads-down building?"
- "What ServiceNow certifications does the team typically pursue?"

### Team and Culture Questions
- "How does the team handle knowledge sharing — are there internal demos, retros, or documentation from past engagements?"
- "What happened with a recent engagement that didn't go as planned? How did the team handle it?"
- "How does the travel actually work — is it predictable or last-minute?"

### Strategic Questions
- "Where do you see the Intelligent Agent Crew in 12 months? Which new agents are prioritized?"
- "How do you differentiate against the Big 4 consultancies when they pitch similar AI solutions on ServiceNow?"
- "The HFS Research study showed 55% of executives are open to switching implementation partners — how is newRocket capitalizing on that?"
