# 02 -- SE (Solutions Engineer) Discovery, Demos, and Positioning for Okta/Auth0

Identity is a technical domain with strong opinions. Customers come in with existing tools, real security incidents, and compliance mandates. Discovery in identity deals requires you to unpack their current state — the protocols they use, the apps they manage, the pain they feel — before you can position a solution. This file covers the discovery framework, demo structure, competitive positioning, and objection handling specific to Okta/Auth0 SE work.

---

## Identity Discovery Framework

### The Four Dimensions of Identity Discovery

Before any Okta or Auth0 conversation, you need answers across four dimensions. Drive toward all four in a discovery call.

| Dimension | What You Are Learning | Key Questions |
|-----------|----------------------|---------------|
| **Current State** | What tools and processes exist today | "Walk me through how a new employee gets access to the systems they need on day one." |
| **Pain** | Where the current state breaks down | "Where does that process break down? What falls through the cracks?" |
| **Business Impact** | Why the pain matters in dollar terms or risk terms | "When access isn't removed on time, what's the exposure?" |
| **Future State** | What success looks like for them | "What does good look like for you a year from now?" |

### Discovery Question Bank: Workforce Identity

These are your open-ended starter questions. Each one should generate 5+ minutes of conversation.

**Access Management / SSO**
- "How do your employees access the applications they need today — is there a central portal, or do they go directly to each app?"
- "How many applications are you managing? What's the mix between SaaS and on-premises?"
- "What's the login experience like for your users today — are they re-entering passwords for each app?"
- "What happens when an employee needs access to a new application — how is that provisioned?"

**Identity Lifecycle (the LCM goldmine)**
- "Walk me through your joiners, movers, and leavers process — from offer accepted to day one access."
- "What happens when someone is promoted or changes teams in terms of their access?"
- "How quickly can you guarantee that a terminated employee's access has been fully revoked across all systems?"
- "Have you ever had an incident where access wasn't removed on time? What did that cost you?"

**Security and Compliance**
- "What authentication methods are you using today — passwords, MFA? What's your MFA adoption rate?"
- "Have you had any account compromise incidents in the past 12 months? What was the vector?"
- "What compliance frameworks are you operating under — SOC2, HIPAA, SOX, FedRAMP?"
- "Do you have any access certification or recertification requirements — periodic reviews of who has access to what?"
- "What does 'zero trust' mean for your team, and where are you on that journey?"

**Technical Environment**
- "What's your current identity stack — Active Directory, Azure AD, LDAP, or something else?"
- "Are you a primarily Microsoft shop, or mixed cloud?"
- "What's your HR system of record? Is it Workday, SAP, ADP, something else?"
- "Do you have any on-premises infrastructure that needs to stay on-prem?"
- "What percentage of your apps support SAML or OIDC vs. needing legacy protocols or form-fill?"

**Decision Dynamics**
- "Who else on your team is involved in evaluating identity solutions?"
- "Is there a security team that will need to review this separately?"
- "Are you looking at other vendors? Who's on the short list?"
- "What does your procurement and security review process look like for something like this?"
- "What's driving the timing on this evaluation — is there an event, an audit, or a renewal coming up?"

### Discovery Question Bank: Customer Identity (Auth0)

**Current State**
- "How are users authenticating to your product today — did you build your own auth or are you using a library?"
- "How many active users are you authenticating per month?"
- "What authentication methods do you support today — passwords, social login, passkeys, anything else?"
- "Are you B2C, B2B, or both? If B2B, do your business customers need their own SSO?"

**Pain (the build-vs-buy conversation)**
- "How much engineering time goes into maintaining your current auth system — feature work, security patches, new protocol support?"
- "What's the biggest headache with your current auth setup?"
- "Have you had any security incidents related to authentication — credential stuffing, account takeover?"
- "When a customer asks you to support their corporate SSO, how long does that take you to build and ship?"

**Scale and Growth**
- "What does your user growth look like over the next 12 months?"
- "Are you expanding into new markets — EU, healthcare, finance? Any compliance requirements that auth needs to support?"
- "What new auth capabilities are on your roadmap that you haven't built yet?"

**Developer Experience**
- "Who owns the auth code today — is it a dedicated team, or does every team touch it?"
- "What's your current stack — what languages and frameworks are in play?"
- "How long does it take to add a new authentication method today?"

### Qualifying the Pain: The Identity Pain Hierarchy

Not all pain is equal. Use this hierarchy to assess urgency and deal velocity.

| Pain Level | Signal | Example Statement | Deal Urgency |
|-----------|--------|-------------------|-------------|
| **Critical** | Security incident occurred or compliance deadline | "We had a breach last quarter" / "Audit is in 60 days" | 30-60 days |
| **Operational** | Manual process causing real cost or risk | "It takes IT 3 days to provision a new hire" | 60-90 days |
| **Strategic** | Initiative underway (cloud migration, M&A, zero trust) | "We're migrating off on-prem AD this year" | 90-180 days |
| **Aspirational** | Would be nice, no burning platform | "Eventually we'd like passwordless" | 180+ days or never |

Lead with critical and operational pain in your follow-up communications. Aspirational pain signals you need to create urgency or wait for it to become operational.

---

## Demo Structure: Okta Workforce Identity

A great Okta Workforce demo tells a story of a day in the life — from the perspective of three personas: the end user, the IT admin, and the security team.

### Demo Narrative Arc (45-minute version)

**Opening hook (3 min):** Tie directly to the pain discovered. "In our conversation you mentioned that offboarding takes days and you had an incident where a contractor still had Salesforce access three weeks after their last day. Let me show you what that looks like with Okta."

**Act 1 — The End User Experience (8 min)**
- Log in at `company.okta.com` → single pane of glass showing all assigned apps
- Click Salesforce → instant SSO, no second login
- Click a restricted app → MFA prompt with Okta Verify push
- FastPass demo: device-bound, biometric, no password at all
- **Message:** This is what your employees experience every day — one place, no password fatigue, phishing-resistant.

**Act 2 — Lifecycle Management (12 min)**
- Switch to admin console
- Show the Workday integration — a new hire event flows in
- Show the group assignment rules: "New hires in Engineering get GitHub, Jira, AWS"
- Show automatic app provisioning firing — the Salesforce account gets created, license assigned, profile populated
- Show offboarding: mark user as terminated → all sessions revoked instantly, apps deprovisioned
- **Message:** Zero manual work. Zero lingering access.

**Act 3 — Adaptive MFA and Security Policy (10 min)**
- Show Authentication Policies — rules engine
- "If the user is on a managed device and a trusted network, allow with FastPass only. If unmanaged device, require Okta Verify push. If new country, require hardware key."
- Show ThreatInsight — live risk scoring per login event
- Show a suspicious login blocked in real-time
- **Message:** Okta evaluates every access request, every time. Not just on login — policy is continuous.

**Act 4 — Workflows (7 min)**
- Open Workflows canvas
- Show the "new hire" flow: Workday triggers → Okta creates user → Slack sends welcome message → ServiceNow ticket created for IT setup
- Show the "failed MFA 5 times" flow → creates IT ticket automatically
- **Message:** No custom scripts. No cron jobs. Automation built into the identity platform.

**Close and next steps (5 min)**
- Recap: tie back to each pain point and which feature addressed it
- "We can stand up a trial environment with your apps and let you test the SSO integrations and LCM against a sample of your real use cases. Who on your team would run the evaluation?"

### Demo Tips for Okta

- Always demo in a pre-built demo org — never the production org
- Have at least two browser tabs open: end-user view and admin view. Practice switching cleanly.
- Pre-configure the apps for the customer's stack (Salesforce for a Salesforce shop, Slack, etc.)
- Show the Okta dashboard in your browser with an extension — the SSO experience is the "wow" moment
- If a feature isn't live yet, offer a video or a roadmap slide — don't promise what isn't there

---

## Demo Structure: Auth0 / Customer Identity

Auth0 demos run differently from Workforce demos. Your audience is developers or product leaders, not IT admins. The demo should feel like looking at code and a product at the same time.

### Demo Narrative Arc (30-minute version)

**Opening (2 min):** "You mentioned you're spending engineering cycles maintaining your auth layer. Let me show you what your developers would own vs. what Auth0 handles automatically."

**Act 1 — Universal Login and the Developer Experience (8 min)**
- Show an app using Auth0's hosted login page — brand it with a custom logo and colors
- Show the login flow: user enters email → Auth0 challenges with appropriate factor
- Switch to the Auth0 dashboard — show the Universal Login settings, how to change branding without deploying code
- **Message:** Your engineers configure this in the dashboard. They don't write login UIs. They don't handle password hashing. They don't build MFA flows.

**Act 2 — Social Login and Connections (5 min)**
- Add Google social login in the dashboard — two clicks
- Demonstrate the login page updating to include "Continue with Google"
- Show how to add an enterprise SAML connection (optional: if B2B)
- **Message:** Adding a new login method is configuration, not code. When a customer asks for their corporate SSO, your team enables it in minutes, not weeks.

**Act 3 — Organizations for B2B (if applicable) (8 min)**
- Create an Organization for a fictional customer ("Acme Corp")
- Add a SAML connection pointing to Acme's Okta (or Entra ID)
- Show that users from Acme log in via their corporate IdP, land in the app, and are identified as Acme members
- Show per-org branding
- **Message:** Every customer gets their own SSO experience without any changes to your application code.

**Act 4 — Actions and Custom Logic (5 min)**
- Open the Actions flow for post-login
- Show a simple Action: pull a user's plan level from your billing API and add it as a custom claim to the token
- **Message:** Auth0 is not a black box. Your developers add business logic at the edge of the auth flow — safely, without touching the authentication engine itself.

**Close (2 min):** "We can give you a free trial tenant today, and I can share the quickstart for your stack — it's usually running in under an hour. Who should we send that to?"

---

## Handling Common Objections

### "We already have Entra ID / Active Directory — why do we need Okta?"

**Approach:** Validate, quantify, differentiate.

"Entra is great for Microsoft-native apps, and I would not suggest replacing it for those. The question is really about coverage and complexity outside the Microsoft ecosystem. How many of your apps are Microsoft-native? In most organizations, it's 30-40%. For the other 60-70% — Salesforce, Workday, AWS, GitHub, ServiceNow — Okta has deeper pre-built integrations, more mature lifecycle management connectors, and more flexible policy options. Many of our customers run Okta federated with Entra — best of both."

### "We built our own auth — it works fine."

**Approach:** Acknowledge, then surface the hidden costs and risks.

"Building your own is totally valid for many teams. What I find is that 'works fine' usually means 'handles the basic case.' What starts to strain homegrown auth is when you need to add passkeys, support enterprise SSO for a new B2B customer, pass a SOC2 Type II audit, or respond to a credential stuffing attack. I'd love to understand what's on your roadmap — which of those things are you going to need to build in the next 12 months, and what's the engineering cost estimate?"

### "Your price is too high."

**Approach:** Anchor on total cost, not sticker price.

"I hear that. Let's frame it in terms of total cost. What's the current cost of maintaining your auth system — engineering hours per sprint, security incidents and the response cost, the cost of manual provisioning? And what's the opportunity cost of engineers maintaining auth infrastructure instead of shipping product? Auth0's pricing is based on active users, so we can model the exact scenario for your user volume. In most cases, the break-even is 6-12 months, and then it's pure engineering savings."

### "We need it to run on-premises / in our own cloud."

**Approach:** Understand the real requirement, then address options.

"Tell me more about that requirement — is this about data sovereignty, regulatory compliance, network isolation, or something else? Understanding the driver helps me present the right options." Then: Okta does not offer on-premises deployment. If the requirement is truly on-premises and non-negotiable, that is a disqualifier for Okta Workforce. If the requirement is data residency, Okta's EU cell may address it. If it's regulatory (FedRAMP), Okta's FedRAMP environment covers it. Be honest — do not try to fit a square peg.

### "What about vendor lock-in?"

**Approach:** Address directly, then reframe.

"That is a fair concern and I want to be honest with you. Any identity platform creates some degree of coupling — your apps integrate against our APIs, your users live in our directory. That is true of any IdP you choose. The question is: what is the cost of migration if you needed to leave, versus the cost of building and maintaining this yourself forever? Okta uses open standards — OIDC, SAML, SCIM — so your apps would work with any compliant IdP. We can export your user data. The lock-in is less about Okta specifically and more about the decision to outsource auth at all."

### "Auth0 vs. building with an open-source library like Passport.js or next-auth"

**Approach:** Scope the comparison correctly.

"Libraries are great for handling the protocol mechanics — token parsing, OAuth flows. What they don't give you is the operational layer: hosted login UI, MFA, bot detection, breached password detection, audit logs, compliance certifications, a managed secret store, universal session management. You would build all of that yourself on top of the library. Auth0 is what you get when you let someone else run those 10 systems as a service. The build-vs-buy question is really: does your team want to be in the auth infrastructure business long-term?"

---

## POC (Proof of Concept) Design for Identity Deals

### Okta Workforce POC Checklist

A typical Okta Workforce POC should answer the customer's top 2-3 integration questions. Do not scope a POC that covers everything — it never finishes.

**Scope template:**
1. **SSO integration for [2-3 named apps]** — demonstrate login, attribute mapping, and app assignment by group
2. **Lifecycle management** — one provisioning flow from the customer's HR system or a simulated CSV import → automatic app provisioning
3. **MFA policy** — demonstrate one Adaptive MFA policy rule that enforces step-up for a sensitive application

**Success criteria to define upfront:**
- "SSO is working for all three applications and users can access them from the Okta dashboard"
- "When a new user is created in [HR system], the Okta account and app assignments are created within 5 minutes"
- "MFA prompts users accessing [sensitive app] from unmanaged devices but not from managed corporate devices"

**POC environment:** Always use a trial org, not the customer's production environment. Okta provides trial orgs with 30-day free access. Request one from your Okta SE team.

### Auth0 / CIC POC Checklist

Auth0 POCs are typically developer-led. The SE's role is to scope the success criteria, unblock technical blockers, and own the presentation.

**Scope template:**
1. **Universal Login with brand customization** — the customer's login page, not Auth0's default
2. **Primary connection** — social login (Google/Apple) and/or database connection for existing users, OR enterprise connection for a B2B scenario
3. **Token validation** — an API that validates Auth0 access tokens and returns the correct scoped response
4. **One Action** — custom claim or custom logic demonstrating extensibility

**Success criteria:**
- "Users can register and log in using [specified methods] and the login page matches our branding"
- "The existing user database is importable or connectable without requiring all users to reset passwords"
- "Our API accepts and validates Auth0 tokens and responds correctly to scoped requests"
- "Custom user attributes from our database appear as claims in the ID token"

**Common POC gotcha:** Custom database connections require the customer to write scripts (Login, Get User, Create User, etc.) that Auth0 calls. Budget time for this — it is the most common POC blocker.

---

## Technical Architecture Conversations

### Identity Architecture for an Enterprise (Workforce)

When a customer asks "how would Okta fit into our environment," draw this on a whiteboard:

```
[HR System: Workday]
        ↓ SCIM
[Okta Universal Directory] ← profile mastering ← [Active Directory]
        ↓
[Authentication Policies + Adaptive MFA]
        ↓
[Application Network]
    ├── Salesforce (SAML)
    ├── ServiceNow (SAML)
    ├── AWS Console (OIDC)
    ├── GitHub Enterprise (SAML)
    ├── Custom Internal App (OIDC)
    └── On-prem apps (via Okta Access Gateway or RADIUS (Remote Authentication Dial-In User Service))
        ↓
[Workflows] → Slack / ServiceNow / Jira
        ↓
[OIG: Access Requests + Certifications]
```

### Identity Architecture for a B2B SaaS Product (Auth0)

```
[Your App: SPA/Mobile/Server-side]
        ↓ redirect
[Auth0 Universal Login]
        ↓ connection routing
    ├── Your Users → Database Connection (Auth0-managed)
    ├── Customer A → SAML (Okta IdP)
    ├── Customer B → OIDC (Azure AD)
    └── Customer C → Social (Google Workspace)
        ↓
[Actions: enrich token, enforce policy]
        ↓
[Tokens issued]
        ↓
[Your API: validates JWT with Auth0 JWKS endpoint]
```

### Sensitive Architecture Questions to Be Ready For

- **"How does Okta handle Active Directory — do we need to replace it?"** No. Okta AD Agent is installed on-premises and syncs users bidirectionally. Okta federates on top of AD; it does not replace it.
- **"Where does Auth0 store user passwords?"** Auth0 stores bcrypt-hashed passwords in their own managed database — they never see plaintext. For custom database connections, passwords stay in the customer's own database; Auth0 never migrates them.
- **"Does Okta support just-in-time provisioning?"** Yes — JIT provisioning on SAML flows creates the Okta user on first login from a federated IdP, without pre-provisioning via SCIM.
- **"Can Okta handle MFA for SSH/server access?"** Yes — Okta Advanced Server Access (now called Okta Privileged Access) handles SSH and RDP access with Okta-authenticated sessions and short-lived certs.
