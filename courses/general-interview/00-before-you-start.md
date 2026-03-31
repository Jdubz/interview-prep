# Module 00: Before You Start

Before diving into interview strategy and behavioral prep, take stock of where you are. This module helps you assess your strengths, understand the interview landscape at the senior/staff level, and build the raw materials that every other module will draw from.

---

## 1. The Senior/Staff Interview Landscape

### What Changes at Senior+ Level

At junior/mid levels, interviews test "can you code?" At senior/staff levels, the bar shifts:

| Dimension | Junior/Mid | Senior (L5/E5) | Staff (L6/E6) |
|-----------|-----------|-----------------|----------------|
| **Coding** | Solve the problem | Solve it cleanly, discuss trade-offs | Design the solution architecture, then implement |
| **System Design** | Not expected | Design a system, justify choices | Design for scale, identify failure modes, drive trade-offs |
| **Behavioral** | "Tell me about yourself" | Leadership, conflict, mentorship stories | Org-level impact, cross-team influence, ambiguity navigation |
| **Communication** | Explain your code | Explain your decisions | Influence without authority, align stakeholders |
| **Bar** | Potential | Proven delivery | Multiplier effect on the team/org |

### The Typical Hiring Loop

```
1. Recruiter Screen (30 min)
   - Fit check, salary range, timeline, role details
   - Your goal: qualify the opportunity, don't oversell

2. Hiring Manager Screen (45 min)
   - Technical discussion + behavioral
   - Your goal: demonstrate senior-level thinking, ask smart questions

3. Technical Phone Screen (60 min)
   - Live coding OR system design (depends on company)
   - Your goal: communicate clearly while solving

4. On-Site / Virtual Loop (4-6 hours)
   Usually includes:
   - Coding Round 1: algorithms/data structures
   - Coding Round 2: practical/applied coding
   - System Design: design a system end-to-end
   - Behavioral: 2-3 interviewers probing leadership stories
   - Team Match / Culture Fit: casual conversation with potential teammates

5. Debrief (internal — you're not there)
   - Each interviewer submits written feedback
   - Hiring committee decides: hire / no-hire / borderline

6. Offer & Negotiation
```

Not every company follows this exactly. Startups may have 2-3 rounds. Big tech may have 5-6. But the components are consistent.

### What Interviewers Actually Evaluate

Every interviewer fills out a scorecard. Here's what they're typically rating:

```
Technical Ability
  - Can you solve problems correctly?
  - Do you understand the fundamentals?
  - Can you reason about trade-offs?

Problem-Solving Approach
  - Do you clarify before diving in?
  - Do you break the problem down?
  - Do you consider edge cases?

Communication
  - Can you explain your thinking clearly?
  - Do you check in with the interviewer?
  - Can you adjust your explanation to the audience?

Leadership & Impact
  - Have you driven projects to completion?
  - Have you navigated ambiguity and made decisions?
  - Have you grown other engineers?

Culture Fit / Values Alignment
  - Do you match the company's operating style?
  - Would people want to work with you?
  - Do you show intellectual humility?
```

---

## 2. Self-Assessment

Before preparing, know where you're strong and where you need work. Be honest — the point isn't to feel good, it's to focus your limited prep time.

### Technical Skills Inventory

Rate yourself 1-5 on each. Focus your prep on 2s and 3s (the areas where study has the highest ROI):

**Core Engineering**
- [ ] Data structures & algorithms (arrays, trees, graphs, hash maps, dynamic programming)
- [ ] System design (databases, caching, queues, load balancing, scaling)
- [ ] API design (REST, GraphQL, versioning, error handling)
- [ ] Database design (schema, indexing, query optimization, migrations)
- [ ] Testing (unit, integration, e2e, test strategy)

**Your Primary Stack** (e.g., TypeScript/Node.js, Python, Go)
- [ ] Language fluency (can you write idiomatic code without looking things up?)
- [ ] Framework mastery (Express/FastAPI/Gin — deep knowledge, not just usage)
- [ ] Runtime internals (event loop, memory model, concurrency model)
- [ ] Production patterns (logging, monitoring, deployment, error handling)

**Infrastructure & Operations**
- [ ] Docker & containers
- [ ] CI/CD pipelines
- [ ] Cloud services (AWS/GCP/Azure)
- [ ] Observability (logging, metrics, tracing)
- [ ] Security fundamentals (auth, encryption, OWASP)

### Behavioral Story Inventory

List 8-12 stories from your career that cover these themes. Each story should have measurable impact.

```
Theme                              Story? (Y/N)    Strength?
─────────────────────────────────────────────────────────────
Led a project to completion         ___             ___
Navigated ambiguity / unclear reqs  ___             ___
Resolved a conflict                 ___             ___
Mentored / grew another engineer    ___             ___
Made a difficult technical decision ___             ___
Failed and learned from it          ___             ___
Influenced without authority        ___             ___
Shipped under pressure / deadline   ___             ___
Improved a process or system        ___             ___
Handled disagreement with manager   ___             ___
Dealt with a production incident    ___             ___
Made a trade-off you weren't sure   ___             ___
about
```

Don't have 8+ stories? That's your first prep task — dig through your career history and find them.

### Identifying Your Gaps

```
Strong areas (4-5 rating):
  → Maintain. Quick refresh before interviews.
  → These are your confidence anchors.

Growth areas (2-3 rating):
  → Focus here. This is where study moves the needle.
  → Use the relevant courses in this repo.

Weak areas (1 rating):
  → Don't try to become an expert. Know enough to not embarrass yourself.
  → Have an honest answer: "I haven't worked deeply with X, but here's how I'd approach it..."
```

---

## 3. Building Your Narrative

### Your Career Story (2-Minute Version)

You will be asked "tell me about yourself" in almost every interview. Have a polished 2-minute narrative:

```
Structure:
1. Where you started and what shaped your engineering philosophy (15 seconds)
2. Key career arc — 2-3 major chapters (60 seconds)
3. What you're looking for now and why this role (30 seconds)
4. What you bring to this specific team (15 seconds)
```

Guidelines:
- **Chronological but selective** — skip irrelevant early career details
- **Impact-focused** — mention outcomes, not just responsibilities
- **Forward-looking** — end with what you want to do next, connecting to this role
- **Practiced but not robotic** — it should sound natural, not memorized

### Your "Superpower"

What's the one thing you're best at? This should be a recurring theme in your stories.

Examples:
- "I take ambiguous problems and turn them into clear, shippable plans"
- "I build reliable systems that don't page people at 3am"
- "I make teams faster by removing technical and process bottlenecks"
- "I bridge the gap between product and engineering"

Your superpower isn't a technology — it's an engineering capability that transcends any specific stack.

---

## 4. Company Research Framework

Before every interview, spend 30-60 minutes researching:

### What to Research

```
Product
  - What does the company actually build? Who uses it?
  - What are the core technical challenges? (scale, real-time, reliability)
  - Recent product launches or pivots

Engineering
  - What's the tech stack? (check job postings, engineering blog, GitHub)
  - What scale do they operate at? (users, requests, data volume)
  - Engineering blog posts — what problems are they solving?
  - Open source contributions

Culture
  - Company values (and whether they seem genuine vs performative)
  - Glassdoor / Blind reviews (filter for signal, ignore noise)
  - Interview process reviews on Glassdoor

Business
  - Revenue model (B2B, B2C, marketplace, enterprise)
  - Competitors and differentiation
  - Recent funding, growth stage, profitability
  - Recent news (layoffs, acquisitions, leadership changes)
```

### Why Research Matters

1. **Tailored stories**: you can connect your experience to their problems
2. **Better questions**: "I saw your engineering blog post about migrating to Kubernetes — what drove that decision?" shows genuine interest
3. **Red flag detection**: you can decide if this is a place you actually want to work
4. **Salary negotiation**: understanding the business helps you negotiate from strength

---

## 5. Interview Logistics

### Pipeline Management

If you're job searching actively, run multiple processes in parallel:

```
Week 1-2: Research + applications (target 10-15 companies)
Week 3-4: Recruiter screens + initial phone screens
Week 5-6: On-sites (try to cluster these — you want competing offers)
Week 7:   Offers + negotiation

Tips:
  - Track everything in a spreadsheet (company, stage, next step, deadline)
  - Time your applications so on-sites overlap within a 1-2 week window
  - Having multiple offers is the single biggest leverage in negotiation
  - It's okay to tell a recruiter "I'm in process with other companies
    and expect to have offers in 2 weeks. Can we align timing?"
```

### Mental Health

Interviewing is draining. Set sustainable limits:

- Maximum 2 on-site loops per week
- Take rest days between intensive rounds
- Rejections are data points, not verdicts on your worth
- After a bad interview, do a 10-minute retrospective, then move on
- Celebrate progress (getting to on-sites, getting positive signals), not just offers

---

## 6. Managing Nerves and Performance Under Pressure

Many experienced engineers fail interviews not because the problems are too hard, but because their brain freezes under pressure. You solve harder problems every day at work — the difference is the artificial stakes and the feeling of being evaluated. This section treats interview anxiety as the concrete, solvable problem that it is.

### Why Your Brain Freezes

When you perceive a threat (and your brain treats "being judged by a stranger" as a threat), your nervous system activates a fight-flight-freeze response. Adrenaline floods your body, your prefrontal cortex (the part that reasons, plans, and writes code) goes partially offline, and your amygdala (the part that screams "DANGER") takes over.

This is not a character flaw. It is a physiological response. And like any physiological response, it can be managed with practice.

```
What's happening:                     What it feels like:
─────────────────────                 ────────────────────
Adrenaline spike                      Heart racing, hands shaking
Prefrontal cortex suppressed          "I can't think straight"
Working memory reduced                Forgetting things you know cold
Tunnel vision / fixation              Getting stuck on one approach
Time distortion                       "I'm running out of time"
```

### Pre-Interview: Lowering Your Baseline

Your anxiety level walking INTO the interview matters more than any in-the-moment trick. These practices lower your baseline so that the adrenaline spike during the interview doesn't push you over the threshold.

**The morning of:**
- Exercise before the interview — even a 20-minute walk. Physical activity metabolizes stress hormones.
- Eat a real meal 60-90 minutes before. Low blood sugar compounds anxiety. Avoid excess caffeine.
- Arrive (or log in) 10 minutes early, not 30. Too early means more time to spiral.
- Do a brief warm-up problem you've already solved. The goal isn't learning — it's getting your brain into "coding mode" so the interview isn't a cold start.

**The night before:**
- Lay out your environment. For virtual: test your IDE, camera, mic, internet. For in-person: know the route, the building, the parking. Eliminating logistics uncertainty reduces background anxiety.
- Stop studying. Cramming the night before adds stress and changes nothing. Review your notes lightly, then do something enjoyable.

**The week before:**
- Practice under simulated pressure (see "Pressure Inoculation" below). The interview should not be the first time you code while nervous.
- Sleep. Chronic sleep deprivation raises cortisol and makes freeze responses more likely. Prioritize 7-8 hours for the 3 nights before an on-site.

### In the Moment: When You Feel the Freeze Coming

You will recognize the freeze: your mind goes blank, you stare at the problem, and you feel the urge to stay silent while panic builds internally. Here is what to do.

**1. Talk out loud immediately.**

Silence is the freeze's best friend. The moment you stop talking, the anxiety loop tightens. Force yourself to narrate, even if what you're saying feels obvious or incomplete:

```
"Okay, let me re-read the problem statement..."
"So the input is an array of integers and I need to return..."
"My first instinct is a brute force approach — let me think about why..."
"I'm going to start by considering the simplest case..."
```

This works because speech activates your prefrontal cortex and breaks the freeze loop. It also buys you time — the interviewer hears you thinking, not struggling.

**2. Write something — anything.**

When you can't think of the solution, write what you DO know:

```
# What I know:
# - Input: list of intervals [(start, end), ...]
# - Output: merged intervals, no overlaps
# - Edge cases: empty list, single interval, already sorted
#
# Brute force idea: compare every pair? That's O(n^2)...
# Better: what if I sort by start time first?
```

Writing externalizes your thinking and frees up working memory. It also gives the interviewer a window into your process — which they're evaluating even more than your final answer.

**3. Use a physical reset.**

When anxiety spikes, your breathing becomes shallow. This reduces oxygen to your brain and worsens the cognitive impairment. Do a physiological sigh — two quick inhales through the nose, one long exhale through the mouth. It takes 5 seconds. You can do it while "reading the problem" or "thinking about the approach." One or two of these can meaningfully lower your heart rate in under 30 seconds.

**4. Name the moment, then move past it.**

It is completely acceptable to say to an interviewer:

```
"Let me take a moment to organize my thoughts."
"I want to make sure I'm approaching this clearly — give me 15 seconds."
"I know I've seen this pattern before. Let me think about where it applies."
```

This sounds confident, not weak. Interviewers respect candidates who self-regulate. What they penalize is prolonged silence with no signal of what's happening.

**5. Fall back on process, not brilliance.**

When your brain is foggy, you will not have a flash of insight. Stop waiting for one. Fall back on a mechanical process:

```
For coding problems:
  1. Restate the problem in your own words
  2. Write out 2-3 concrete examples by hand
  3. Identify the brute force approach
  4. Code the brute force
  5. Only then consider optimization

For system design:
  1. List functional requirements
  2. List non-functional requirements
  3. Draw boxes and arrows for the obvious components
  4. Pick one component and go deeper

For behavioral questions:
  1. Pick a story from your bank (you prepared these in section 2)
  2. State the Situation in one sentence
  3. State the Task
  4. Walk through the Action
  5. State the Result
```

Process is anxiety-proof. Brilliance is not. When you have a checklist to follow, your prefrontal cortex has a handrail.

### Pressure Inoculation: Training for the Adrenaline

The single most effective way to reduce interview anxiety is to practice under conditions that trigger it. Your goal is to make the real interview feel familiar, not novel.

**Level 1 — Timed problems (solo)**
- Set a 25-minute timer on LeetCode or a practice problem. When the timer starts, your heart rate should go up slightly. If it doesn't, you're too comfortable — add stakes (e.g., "if I don't solve it, I skip my next break").
- Practice talking out loud while solving, even alone. This feels ridiculous at first. Do it anyway.

**Level 2 — Mock interviews with a friend**
- Ask a colleague or friend to interview you. Use a real problem they choose (not one you've seen). Have them sit silently while you work — the silence is part of the pressure.
- Debrief after: what did you feel? Where did you freeze? What technique did you use (or wish you'd used)?

**Level 3 — Mock interviews with a stranger**
- Use services like Pramp, interviewing.io, or find a stranger through a Discord community. The discomfort of performing for someone you don't know is much closer to a real interview.
- Record these sessions if possible. Watching yourself will reveal habits you don't notice in the moment (long silences, fidgeting, apologizing for your thinking).

**Level 4 — Real interviews at companies you care less about**
- Schedule 2-3 interviews at companies that are not your top choice. Treat them as live practice. The stakes feel real because they ARE real, but the outcome matters less to you.
- This also builds momentum and may produce offers you can use as negotiation leverage.

### Reframing the Interview

The stories you tell yourself about the interview shape your anxiety response. Some reframes that help:

**"This is a conversation, not an exam."**
The interviewer is trying to determine if you'd be a good colleague. They want you to succeed — an empty pipeline costs them time and money. In most cases, they're actively rooting for you.

**"I'm evaluating them too."**
You are not a supplicant begging for a job. You are a senior engineer deciding whether this team, codebase, and company deserve your next few years. The interview is bidirectional. Sitting in this posture changes how you carry yourself.

**"A freeze is a data point, not a verdict."**
If you freeze, it doesn't mean you're not good enough. It means your nervous system did what nervous systems do. Use the recovery techniques above, finish the interview, and debrief after. Some of the best interview performances include a recovery from a rough start.

**"I've solved harder problems than this."**
Before the interview, write down 3 genuinely hard problems you've solved at work. Debugging a production outage at 2am. Designing a system that handles 10x traffic. Mentoring a struggling engineer through a breakthrough. Remind yourself that an interview problem is a toy compared to real engineering.

### The Post-Interview Debrief

After every interview (good or bad), spend 10 minutes writing down:

```
1. Where did I feel confident? Why?
2. Where did I freeze or struggle? What triggered it?
3. What recovery technique did I use (or wish I'd used)?
4. What would I do differently in the next interview?
5. One thing I did well that I want to repeat.
```

This turns every interview into training data. Over time, you'll notice your freeze triggers shrink and your recovery speed improves.

---

## 7. How to Use This Course

### Recommended Study Order

```
1. This module (00) — self-assessment, narrative, logistics
2. Module 02 — Behavioral Interview Mastery (takes the most practice time)
3. Module 01 — Interview Strategy (framework for the whole process)
4. Module 03 — Technical Communication (how you say things matters as much as what you say)
5. Module 06 — Coding Interview Patterns (if you need algorithm refreshing)
6. Module 05 — Common Technical Questions (stack-agnostic technical knowledge)
7. Module 04 — Questions to Ask Interviewers (prepare these in advance)
8. Module 08 — Negotiation & Closing (when you have offers)
```

### Pair With Technical Courses

This course covers the meta-skills. Pair it with the technical courses for your target stack:

| Target Role | Technical Courses |
|-------------|-------------------|
| Full-stack (TS) | Node.js + React + Infrastructure |
| Backend (Python) | Python/FastAPI + Infrastructure |
| Backend (Go) | Golang + Infrastructure |
| AI/ML Engineer | LLMs + Python/FastAPI + Infrastructure |
| Generalist Senior | Node.js + Infrastructure |

### Practice Schedule

```
Daily (30-60 min):
  - Practice one behavioral story out loud (yes, out loud)
  - Solve one coding problem (LeetCode medium, timed)

Weekly (2-3 hours):
  - One mock system design (45 min + review)
  - One pressure inoculation session (mock interview or timed problem with talk-aloud)
  - Review and refine behavioral stories based on research for upcoming interviews

Before each interview:
  - Re-read company research notes
  - Review 3-4 stories tailored to this company's values
  - Prepare 4-5 questions for interviewers
  - Quick-scan relevant cheat sheets from technical courses
```

---

## 8. Quick Checklist

Before moving to Module 01:

- [ ] Completed the technical skills self-assessment (rated 1-5)
- [ ] Identified 8+ behavioral stories from your career
- [ ] Written a 2-minute career narrative
- [ ] Identified your "superpower"
- [ ] Know the typical interview loop structure
- [ ] Identified your freeze triggers and chosen 2-3 recovery techniques to practice
- [ ] Scheduled at least one mock interview for pressure inoculation
- [ ] Started a tracking spreadsheet (if actively searching)

---

## Next Steps

Start with [Module 02: Behavioral Interview Mastery](02-behavioral-mastery/) — building your story bank takes the most practice time, so start early. Then come back to [Module 01: Interview Strategy](01-interview-strategy/) for the tactical framework.

---

## Practice

- Complete the self-assessment checklists above before moving to any other module. Be honest -- the value of this exercise is in identifying gaps, not feeling good.
- Write out your 2-minute career narrative and practice delivering it out loud three times. Record yourself on the third attempt and listen back.
- Draft your "superpower" statement. Test it with a trusted colleague: does it ring true? Would they describe you the same way?
- Fill in the company research template for one target company. Time yourself -- aim for 30-45 minutes.
- Do one timed coding problem (25 minutes) while narrating out loud. Notice where anxiety spikes and practice at least one recovery technique (physiological sigh, writing what you know, falling back on the step-by-step process).

---

## Cross-References

- **[Module 01 — Interview Strategy](./01-strategy/):** Expands on the pipeline management and interview logistics introduced here. Start with Module 00 for self-assessment, then move to Module 01 for tactical execution.
- **[Module 02 — Behavioral Interview Mastery](./02-behavioral/):** The behavioral story inventory in this module feeds directly into Module 02's story bank. Complete your inventory here first, then use Module 02 to refine each story into STAR format.
- **[Module 03 — Technical Communication](./03-technical-communication/):** Your career narrative and "superpower" statement are communication exercises. Module 03 covers the broader communication skills you will use in every interview round.
- **[Module 08 — Negotiation & Closing](./08-negotiation/):** The company research framework here (business model, revenue, competitors) feeds directly into negotiation preparation. Understanding the business helps you negotiate from a position of knowledge.
