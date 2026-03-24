# Story Flex Guide — Use Any Story for Any Question

You have 7 distinct stories from your career. Each can be angled for
multiple prompts. Don't lock a story to one question — know the flex.

## Your Story Inventory

| # | Story | Primary Angle | Where Used |
|---|---|---|---|
| A | Amazon Fresh integration | Technical depth, cross-company collaboration | Ali — project deep dive |
| B | Unified ordering abstraction | Enabling others, architecture trade-offs | Ali — project 2 |
| C | Test order generator (HOG) | Users first, questioned the spec | Tyler — story 1 |
| D | Order management UI rewrite | Proactive, self-assigned, DX improvement | Tyler — story 2 |
| E | Structured logging refactor | Curiosity, went beyond the ask | Tyler — story 5 |
| F | Blinky Time TinyML | Deep curiosity, learning for its own sake | Tyler — backup |
| G | Opna Development (consulting) | Client-facing, managing relationships | Unassigned |

## Stories 3, 4, 6 are EMPTY in Tyler's worksheet

You need to fill these gaps. Here's how to use what you already have:

---

### Story 3 (Learning from a user interaction) — USE: Amazon Fresh spec ambiguity

You already have this material in your Ali deep dive but it's a DIFFERENT
ANGLE. For Ali, you talk about the technical architecture. For Tyler,
you talk about what you learned about COMMUNICATION.

```
Situation: Building the AZ event pipeline, we were both building to
the same 20-page spec simultaneously.

Task: I was implementing hundreds of pub/sub events and needed to
ensure our interpretation matched theirs.

Action: Early on, I discovered that a requirement we read as "report
item status on every state change" meant something different to each
team. We reported the status of the item in our system. They expected
the status of the physical item in the real world (e.g., "broken",
"missing"). I set up weekly alignment calls to walk through the spec
line by line with their engineering team, comparing interpretations.

Result: We caught 12+ misalignment issues before they became
production bugs. The process became the template for how Fulfil
onboarded future partners.

What assumption did this break?
I assumed that a detailed spec meant we had a shared understanding.
I learned that written requirements are necessary but NOT sufficient
— you need synchronous alignment with the humans interpreting them.
```

**Why this works for Tyler:** It's about learning from an interaction
with a stakeholder (AZ's team), not about the architecture.

**Why it doesn't overlap with Ali:** For Ali, you talk about pub/sub,
event pipeline, DB scaling. For Tyler, you talk about the communication
failure and what you learned about specs.

---

### Story 4 (Balancing user needs vs constraints) — USE: Marketplace unified ordering

You have this in Ali's worksheet as Project 2, but again DIFFERENT ANGLE.

```
Situation: We were unifying 3 marketplace pipelines (Fulfil, Uber Eats,
DoorDash). Each partner wanted their pipeline to work exactly like their
existing API. Our operators wanted one dashboard for all sources.

Task: Balance each partner's unique requirements against the operators'
need for a single, consistent interface.

Action: I discovered the partners didn't actually need the pipeline to
work identically to their existing APIs — they needed their ORDER DATA
to be accurate. The operators needed consistent STATUS FIELDS. So I
designed a polymorphic schema that stored the original partner data
in source-specific tables (partners happy) while normalizing the
status fields that operators needed for the dashboard (operators happy).

Result: 3 operators → 1 operator for standard volume. New AZ source
onboarded in days instead of weeks.

How did you figure out what they actually needed vs what they asked for?
The partners asked for "our API format." What they actually needed was
"our data, accessible and correct." The operators asked for "one
dashboard." What they actually needed was "consistent fields to filter
and search by." I found this by asking "what do you do with this data
after you get it?"
```

**Why this works for Tyler:** It's about understanding what users
ACTUALLY need vs what they SAY they need. Classic Users First.

**Why it doesn't overlap with Ali:** For Ali, you talk about the
abstraction layer architecture and technical trade-offs. For Tyler,
you talk about user research and the gap between requests and needs.

---

### Story 6 (Changed approach based on new information) — USE: HOG test order generator

Story C (test order generator) already HAS this arc built in. You
just angle it differently.

```
Situation: I was building the test order generator from an existing spec.

Task: Build what the spec described — a cron-based test order system.

Action: After starting to implement the spec, I realized it would
inherit the same limitations. I CHANGED my approach: instead of
building to spec, I interviewed the robotics team. I discovered they
needed location-based ordering, not category-based. I pivoted
completely to a new design.

Result: Eliminated the 90% capacity throughput ceiling. Doubled
system throughput.

What made you willing to change course?
I realized I was about to spend 2 weeks building something I already
knew wouldn't work. The spec described the WRONG tool because the
people who wrote it didn't understand the real bottleneck. Once I
talked to the actual users, the path was obvious.
```

**Why this is different from Story 1 (Tyler):** Story 1 emphasizes
going to the user and discovering their real need. Story 6 emphasizes
the moment of changing your approach — the willingness to abandon
a spec you were given.

---

## The Flex Matrix — Which story answers which question

| Question | Best Story | Backup Story |
|---|---|---|
| **Ali: Technically challenging project** | A (Amazon) | B (Unified ordering) |
| **Ali: Enabled others to succeed** | B (Unified ordering) | E (Logging refactor) |
| **Ali: Cross-functional project** | A (Amazon) | B (Unified ordering) |
| **Ali: Disagreed with a team decision** | B (Queue job pattern — you'd change it) | E (Advocating for structured logging) |
| **Ali: Hardest bug** | A (Event reconciliation) | D (Tracing 2 conflated data systems) |
| **Ali: Mistake you learned from** | B (Queue job over-abstraction) | A (Monolith scaling) |
| **Tyler: Designed from user perspective** | C (HOG — interviewed robotics team) | D (Order UI — separated data systems) |
| **Tyler: Proactive UX/DX fix** | D (Order UI — self-assigned) | E (Logging — went beyond the ask) |
| **Tyler: Learned from user interaction** | A-angle (Spec misalignment with AZ) | C (Robotics team didn't know what to ask for) |
| **Tyler: Balanced user needs vs constraints** | B-angle (Partners vs operators) | A (AZ requirements vs system limitations) |
| **Tyler: Pure curiosity** | E (Logging refactor) | F (Blinky Time TinyML) |
| **Tyler: Changed approach** | C-angle (Abandoned the spec) | E (Discovered logging was the root cause) |

## Additional Stories From Your Background (unused)

These are in your database but NOT in any worksheet. Have them ready
as backup if a question doesn't fit your prepared stories:

### Dialogflow / JLL — Microservice behind NLP chatbot
- **Angle for Ali:** Complex architecture — microservices behind NLP,
  handling everything from meeting scheduling to facility requests
- **Angle for Tyler (Curiosity):** "The most requested feature turned
  out to be temperature control. Nobody expected that. We had built
  this sophisticated NLP system and the #1 use case was 'it's cold
  in conference room 3.'"

### Britelite — Live broadcast software for red carpet events
- **Angle for Tyler (Users First):** Zero-tolerance, live-on-air
  software. The user is a celebrity on camera. Ghost touches from
  IR flood lights. You can't deploy a fix — it's live.
- **Angle for Ali (Pressure):** "millions of viewers, no room for
  error, calibrating for camera color temperature while the show
  is running"

### Opna — Co-founder, client-facing consulting
- **Angle for Ali (Leadership):** "I was the solution architect
  for all new contracts, managed client relationships and timelines"
- **Angle for Tyler (Users First):** Client discovery — pitching
  requires understanding what the client actually needs, not just
  what they say in the RFP

### Meow Wolf — Interactive installation pub/sub
- **Angle for Tyler (Curiosity):** Working on interactive art
  installations, puzzle completion tracking, fleet management
- Quick mention only — thin on detail

### Type 1 Diabetes — Personal resilience
- **Angle for Ali (What drives you):** Not a work story but a
  powerful one-liner for "what makes you who you are" questions:
  "I've been managing a life-threatening condition since I was 18
  months old. It taught me to stay calm where others panic and to
  make sound decisions under pressure."
- Use sparingly — only if they ask about resilience or pressure.

## Rules for Winging It

1. **Know the ANGLE, not the script.** Every story has 2-3 angles.
   Pick the one that answers THIS question.

2. **If a question fits two stories, use the one you haven't told yet.**
   Tyler and Ali compare notes — never repeat.

3. **If no prepared story fits, use the Britelite or Dialogflow backup.**
   These are rich enough to wing but you haven't over-prepared them,
   so they'll sound authentic.

4. **If you blank, buy time with the restate technique:**
   "That's a great question. Let me think about the best example...
   [5 seconds] ...actually, there's a situation from when I was
   building the [X] that's a perfect fit."

5. **Short > long.** A 90-second story with a clear point beats
   a 3-minute ramble. The interviewer will ask follow-ups if they
   want more.
