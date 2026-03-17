# 01 -- Okta Platform and Product Knowledge

You cannot win an SE interview at Okta without deep, fluent product knowledge. This is not a generic IAM (Identity and Access Management) concepts review. This is the exact knowledge you need to speak credibly about Okta's platform on day one of an interview loop — from the product architecture, through the key protocols, to how the two product lines (Workforce Identity and Customer Identity) differ and when each applies.

---

## The Okta + Auth0 Landscape

Okta acquired Auth0 in May 2021 for $6.5B. The two brands still exist and serve different buyer personas. Understanding the distinction is essential — interviewers will probe this.

### Product Line Overview

| Product | Brand | Primary Buyer | Core Problem Solved |
|---------|-------|---------------|---------------------|
| **Okta Workforce Identity Cloud** | Okta | IT, Security, HR | Employee SSO (Single Sign-On), lifecycle management, zero trust for internal apps and infrastructure |
| **Okta Customer Identity Cloud (CIC)** | Auth0 (powered by Auth0) | App developers, product teams | Customer-facing authentication — login, registration, MFA (Multi-Factor Authentication), social login — embedded into a product |
| **Okta Identity Governance (OIG)** | Okta | IAM teams, compliance | Access requests, certifications, entitlement management, separation of duties |
| **Okta Privileged Access** | Okta | Security, DevSecOps (Development, Security, and Operations) | Privileged account management, SSH (Secure Shell)/RDP (Remote Desktop Protocol) access, just-in-time admin elevation |

**The one-sentence distinction:** Workforce is for your employees accessing company apps. Customer Identity (Auth0) is for your customers logging into your product.

### Who Buys Which

- **CISO (Chief Information Security Officer)/IT Director** → Workforce Identity. Cares about SSO, MFA enforcement, phishing-resistant auth, lifecycle management (join/move/leave), compliance reporting.
- **CTO (Chief Technology Officer)/VP Engineering** → Customer Identity (Auth0). Cares about auth developer experience, login UX (User Experience), SDK (Software Development Kit) quality, CIAM (Customer Identity and Access Management) at scale, B2B (Business-to-Business) multi-tenancy.
- **Product Manager** → Customer Identity. Cares about sign-up conversion, social login, passwordless UX.
- **IAM Team** → Both. Workforce for internal, may also own CIC if the company builds customer-facing products.
- **Compliance/GRC (Governance, Risk, and Compliance)** → Workforce + OIG. Cares about access reviews, policy enforcement, audit trails.

---

## Identity Protocol Foundations

You must be able to explain these clearly to both technical and non-technical audiences. Know the difference, know when each applies, and know where Okta/Auth0 supports each.

### OAuth 2.0 (Open Authorization)

An authorization framework. It answers: **"What can this application do on behalf of this user?"**

- Grants an application an **access token** scoped to specific permissions (scopes)
- Does **not** tell you who the user is — that is OIDC's job
- Flows: Authorization Code (web apps), Authorization Code + PKCE (Proof Key for Code Exchange) (SPAs (Single Page Applications), mobile), Client Credentials (machine-to-machine), Device Code (TV/CLI (Command Line Interface) flows)
- Okta acts as the Authorization Server. Client apps redirect to Okta, user authenticates, Okta issues tokens.

**The PKCE explanation for non-technical stakeholders:** "PKCE is like a secret handshake that a mobile app sets up before it asks for a token, so even if someone intercepts the exchange, they cannot use it. It is how we secure OAuth for apps that cannot store a client secret safely."

### OpenID Connect (OIDC)

An identity layer on top of OAuth 2.0. It answers: **"Who is this user?"**

- Adds an **ID token** (JWT (JSON Web Token)) containing user identity claims (`sub`, `email`, `name`, `groups`)
- Adds the `/userinfo` endpoint for fetching additional claims
- Is the modern standard for SSO in web and mobile apps
- Okta fully supports OIDC as an Identity Provider (IdP)

**Key tokens:**
- **ID Token** — identity assertion, consumed by the client application. Do not use this to call APIs.
- **Access Token** — authorization grant, sent to resource servers (APIs). Okta access tokens can be opaque or JWT format.
- **Refresh Token** — long-lived token used to get new access tokens without re-authentication.

### SAML 2.0 (Security Assertion Markup Language)

An XML-based federation protocol. Answers: **"I know who this user is, here is an assertion."**

- Still dominant in enterprise SSO for legacy apps and business applications (Salesforce, Workday, ServiceNow, etc.)
- IdP-initiated vs SP (Service Provider)-initiated flows matter for customer conversations
- Okta was built on SAML — it has the most robust SAML SP and IdP support in the market
- Auth0/CIC also supports SAML but is more commonly used with OIDC

**When to recommend SAML vs OIDC:**
- SAML: Existing enterprise application that only speaks SAML, legacy app, business software integrations
- OIDC: New applications, modern SaaS, mobile apps, SPAs — default recommendation

### SCIM (System for Cross-domain Identity Management)

Protocol for automated user provisioning and deprovisioning between systems.

- SCIM 2.0 is the standard. REST (Representational State Transfer)-based, JSON (JavaScript Object Notation) payloads.
- Enables: creating users automatically in downstream apps when onboarded, updating attributes, deprovisioning access when offboarded
- Okta is the SCIM Client (pushes changes) when provisioning to apps like Salesforce, Slack, GitHub
- Okta can also be a SCIM Server (receives pushes) from HR systems like Workday or BambooHR

**Why this matters in discovery:** When a prospect says "We spend hours every week manually creating and removing user accounts," that is a SCIM conversation. SCIM-enabled provisioning solves this entirely.

### WebAuthn / Passkeys

The modern phishing-resistant authenticator standard.

- W3C (World Wide Web Consortium) standard backed by FIDO2 (Fast Identity Online 2). Replaces passwords and TOTP (Time-based One-Time Password) with public-key cryptography.
- User registers a credential (biometric, hardware key) bound to the origin (domain). Phishing is impossible because credentials are domain-bound.
- Okta FastPass + Okta Verify implement passkey-grade authentication for workforce
- Auth0/CIC supports passkeys for customer-facing apps
- **Positioning:** When a customer has phishing incidents or is pursuing zero trust, lead with WebAuthn/passkeys.

---

## Okta Workforce Identity: Key Capabilities

### Universal Directory

Okta's cloud-hosted LDAP (Lightweight Directory Access Protocol)/user store. The central source of truth for identity.

- Can act as a standalone directory (replacing AD (Active Directory) for cloud-first orgs) or as a profile master / aggregator above AD, LDAP, HR systems
- Supports custom attributes, group management, profile sourcing from multiple systems
- **Discovery signal:** "We have five different user stores that don't sync." → Universal Directory + profile mastering conversation.

### Single Sign-On (SSO)

- The Okta Integration Network (OIN) has 7,000+ pre-built integrations — every major SaaS app has a connector
- SAML and OIDC integrations; Okta also supports Bookmark Apps, SWA (Secure Web Authentication / form-fill as last resort), and WS-Federation (Web Services Federation)
- The "Application Network" differentiates Okta from competitors who have fewer pre-built integrations

### Lifecycle Management (LCM)

- Automates the joiner/mover/leaver process: user created in HR → Okta auto-provisions access to all apps based on group membership and rules
- Integrates with Workday, SAP SuccessFactors, BambooHR, ADP, and others as HR sources of truth
- **Common pain addressed:** "When someone leaves, it takes us days to remove all their access." → LCM with immediate deprovisioning.

### Adaptive Multi-Factor Authentication

- Okta Verify (push, TOTP, FastPass), SMS, email OTP (One-Time Password), hardware tokens (YubiKey/FIDO2), WebAuthn
- Adaptive policies: step-up based on risk signals — new device, new location, anomalous behavior, ThreatInsight risk score
- **Policy engine:** Authentication Policies with rules combining user context, device context, network zone, and risk

### Okta Workflows

- No-code/low-code automation platform built into Okta
- Pre-built connector library for Slack, Salesforce, ServiceNow, Jira, etc.
- Use cases: auto-assign groups on new hire, send Slack notification on offboarding, trigger IT ticket on failed MFA attempts
- **Differentiation:** Competitors require external tools (Zapier, custom scripts) for automation Okta can do natively

### Okta Identity Governance (OIG)

- Access Request: users self-request access, approval workflows route to managers or resource owners
- Access Certification: periodic entitlement reviews ("certifications") where managers confirm access is still appropriate
- Identity Lifecycle policies tied to governance
- **Buyer:** Compliance teams, auditors, companies dealing with SOX (Sarbanes-Oxley Act), HIPAA (Health Insurance Portability and Accountability Act), SOC 2 (System and Organization Controls 2)

### Zero Trust and Device Trust

- Okta Device Trust: requires device to be managed (enrolled in MDM (Mobile Device Management) like Jamf or Intune) before granting access
- FastPass: passwordless, device-bound credential. No password prompt, just biometric on managed device.
- Network zones: allow/deny access from known IP ranges or anonymous proxies
- **Zero trust positioning:** "Never trust, always verify" — Okta evaluates every access request against user, device, location, and behavior signals before granting access.

---

## Auth0 / Okta Customer Identity Cloud: Key Capabilities

### Tenants and Applications

- Everything in Auth0 lives in a **tenant** — an isolated namespace with its own users, apps, and connections
- **Applications** are the things that connect to Auth0 for authentication: web apps, SPAs, APIs, mobile apps, M2M
- Application types determine which flows are available (Regular Web App → Authorization Code; SPA → Auth Code + PKCE; M2M → Client Credentials)

### Universal Login

- Auth0's hosted login page — the central UI through which all authentication flows run
- Can be fully branded (logo, colors, CSS)
- **New Universal Login** (recommended) vs Classic Universal Login — New is built on Auth0's own SDK, more customizable, better MFA support
- **Why it matters:** Auth0 owns the login page, which means the customer's app never handles credentials — a major security and compliance benefit

### Connections

Auth0's term for identity sources — where users come from.

| Connection Type | Examples | Use Case |
|----------------|----------|----------|
| **Database** | Auth0-managed user store | New apps with no existing users |
| **Social** | Google, Apple, Facebook, LinkedIn, GitHub | Consumer apps needing social login |
| **Enterprise** | SAML, OIDC, Active Directory, LDAP, Azure AD | B2B apps where customers bring their own IdP |
| **Passwordless** | Email magic link, SMS OTP | Frictionless consumer login |
| **Custom Database** | Connect Auth0 to existing user DB | Migration or hybrid scenarios |

### Actions

- Auth0's serverless function pipeline for customizing identity flows
- Triggers: Login, Registration, Pre-user-registration, Post-login, etc.
- Replace the older Rules and Hooks (both deprecated, migration required)
- Use cases: enrich ID tokens with custom claims, call external APIs during login, block users based on custom logic, enforce MFA conditionally
- **Key SE talking point:** Auth0 lets developers customize auth without forking the auth system — they add business logic at the edge of the auth flow safely.

### Organizations

- Auth0's B2B multi-tenancy primitive — each "Organization" represents a business customer
- Per-org: enterprise connections (BYOIDP (Bring Your Own Identity Provider)), branding, member management, invitation flows
- **When to lead with Organizations:** The prospect is a B2B SaaS company that needs each of their customers to have their own SSO, their own user directory, and customized login experience. This is Auth0's strongest B2B differentiation.

### Machine-to-Machine (M2M)

- Client Credentials flow: service authenticates directly with Auth0 using a client ID + secret, gets an access token
- Used for backend-to-backend communication, microservices, CLIs, CI/CD (Continuous Integration/Continuous Deployment) pipelines
- Token caching is important — M2M tokens should be cached to rate limit, not requested per call

### Attack Protection

- Breached Password Detection: checks credentials against known breach databases (powered by HaveIBeenPwned)
- Brute Force Protection: lockout after N failed attempts
- Bot Detection: CAPTCHA-based protection on signup/login flows
- Suspicious IP Throttling
- **Security differentiation:** Auth0 embeds security controls directly into the auth flow — no separate WAF (Web Application Firewall) or bot management tool needed for auth-specific threats.

---

## Competitive Landscape

### The Major Competitors

| Competitor | Primary Market | Okta's Differentiation |
|-----------|---------------|----------------------|
| **Microsoft Entra ID** (formerly Azure AD) | Workforce, Microsoft shops | Okta is multi-cloud/multi-IdP neutral; Entra locks you to Microsoft ecosystem; OIN breadth |
| **Ping Identity** | Enterprise workforce | Okta is cloud-native; Ping is historically on-premises and complex to operate |
| **ForgeRock** (now part of Ping) | Enterprise, financial services | Same as Ping — legacy on-prem DNA |
| **AWS Cognito** | CIC/CIAM for AWS-native apps | Auth0 is developer-experience and feature-rich; Cognito is functional but limited |
| **Microsoft Entra External ID** | CIC/B2B, B2C | Auth0 is more feature-complete; Entra External ID still maturing |
| **Clerk** | Developer-focused CIAM, startups | Auth0 scales to enterprise; Clerk is usage-based and newer |
| **Stytch** | Developer-focused CIAM | Same as Clerk — Auth0 has compliance certifications, OIG, enterprise track record |
| **Firebase Authentication** | Consumer mobile apps | Firebase is Google-ecosystem; Auth0 is more enterprise-capable, no vendor lock-in |
| **In-house auth** | Any company that "built their own" | Risk, maintenance burden, compliance liability |

### Entra ID Battle Card (the Most Common Competitive Scenario)

**When Entra ID wins:**
- Customer is all-Microsoft (M365, Azure, Dynamics) and is cost-conscious
- Budget is already inside EA (Enterprise Agreement) agreement
- Basic SSO requirements only — no complex LCM, no non-Microsoft apps

**When Okta wins:**
- Multi-cloud or non-Microsoft primary stack
- Large OIN app catalog needed (7,000+ vs Entra's much smaller gallery)
- Complex LCM with non-Microsoft HR systems (Workday, BambooHR)
- Zero trust across heterogeneous infrastructure (AWS + Azure + on-prem)
- Customer Identity requirements alongside Workforce (unified CIAM + WF)
- Stringent compliance needs: FedRAMP High, SOC 2 Type II, HIPAA (Okta has broad certifications)

**Common Entra objection:** "We already have Entra — why pay for Okta too?"
**Response:** "Entra is great for Microsoft apps. The question is what percentage of your apps are Microsoft-native. For most enterprises, it's 30-40%. For the other 60-70% — Salesforce, ServiceNow, Workday, AWS, custom apps — Okta's OIN catalog and LCM connectors deliver faster time-to-value and more complete coverage. Many of our largest customers run both: Entra for Microsoft apps, Okta as the meta-IdP federating everything."

---

## Key Architecture Patterns

### Okta as Hub (Workforce)

```
HR System (Workday) → [SCIM/API] → Okta Universal Directory
                                           ↓
                              [Profile Mastering + LCM]
                                           ↓
         ┌─────────────────────────────────────────────────────────┐
         │                                                         │
   Salesforce (SAML)   ServiceNow (SAML)   AWS (OIDC)   Custom App (OIDC)
```

Okta is the hub. Every app trusts Okta. Users are provisioned and deprovisioned centrally.

### Auth0 Customer Identity Pattern (B2C (Business-to-Consumer))

```
Your App (SPA/Mobile/Web) → [Redirect] → Auth0 Universal Login
                                               ↓
                              [Connections: Social, DB, Passwordless]
                                               ↓
                              [Actions: enrich token, enforce rules]
                                               ↓
                              [Tokens issued] → App receives ID + Access Token
                                               ↓
                              Your API → [validates Access Token with JWKS (JSON Web Key Set)]
```

### Auth0 B2B Multi-Tenant Pattern (Organizations)

```
Your SaaS App → Auth0 → Organization A → Enterprise Connection (customer's Okta/Entra)
                      → Organization B → Enterprise Connection (customer's Ping)
                      → Organization C → Database Connection (smaller customer, no IdP)
```

Auth0 Organizations lets each of your customers bring their own identity provider without any code changes in your app.

---

## Compliance and Certifications

Knowing these is table stakes for security-conscious buyers.

| Certification | Okta Workforce | Auth0/CIC | Notes |
|--------------|----------------|-----------|-------|
| SOC 2 (System and Organization Controls 2) Type II | Yes | Yes | Both |
| ISO (International Organization for Standardization) 27001 | Yes | Yes | Both |
| HIPAA (Health Insurance Portability and Accountability Act) BAA | Yes | Yes | Business Associate Agreement available |
| FedRAMP (Federal Risk and Authorization Management Program) High | Yes | Limited | Okta FedRAMP is a separate environment; critical for government |
| PCI DSS (Payment Card Industry Data Security Standard) | Yes | Yes | Payment card industry |
| GDPR (General Data Protection Regulation) | Yes | Yes | Data residency: EU Cell available in both |
| CCPA (California Consumer Privacy Act) | Yes | Yes | |
| CJIS (Criminal Justice Information Services) | Yes (FedRAMP env) | No | Law enforcement use cases |

**Data residency:** Okta offers US and EU cells. Auth0 offers US, EU, AU deployments. This comes up in every EU enterprise deal.

---

## Practice Q&A: Product Knowledge

**Q: What is the difference between Okta and Auth0 now that they are the same company?**

They serve different use cases and different buyers. Okta Workforce Identity is built for employee and partner identity — SSO to SaaS apps, lifecycle management, IT-driven security policy. Auth0 (branded as Okta Customer Identity Cloud) is built for developers who need to add login, registration, and authentication to their customer-facing products. The underlying architecture is different, the buyer is different, and the integration model is different. Okta is admin-configured; Auth0 is developer-coded. Both have the full backing of Okta's security certifications and enterprise scale.

**Q: When would you recommend a customer use Auth0 Organizations?**

When a B2B SaaS company needs each of their business customers to have their own SSO, their own user pool, and potentially their own branded login experience. Without Organizations, you would have to manually segregate users and connections per tenant in your application logic. Organizations gives you that multi-tenancy natively in Auth0 — per-org connections, per-org branding, per-org member management — without any changes to your app's core auth code.

**Q: A customer asks why they need Okta when Microsoft Entra is already included in their M365 license. How do you respond?**

I validate the question first — "That's a fair point, Entra handles Microsoft workloads well." Then I quantify the problem: "What percentage of your app portfolio is Microsoft-native?" In most enterprises, it's under 40%. Entra's app gallery has roughly 1,500 pre-built integrations. Okta's OIN has 7,000+. For lifecycle management, Entra requires Azure premium licensing and complex configuration for non-Microsoft HR systems. For zero trust across AWS, GCP, or on-prem infrastructure, Okta's Device Trust and FastPass are more mature. The positioning is not "replace Entra" — it's "Okta federates with Entra for Microsoft apps and covers the rest of your environment."

**Q: Explain SCIM provisioning to a non-technical VP of HR.**

"Right now, when you hire someone, your IT team manually creates accounts in Salesforce, Slack, GitHub, and your internal apps — one by one. And when someone leaves, they have to remember to remove access from every single system. With SCIM, Workday and Okta speak the same language. The moment HR marks someone as hired in Workday, Okta automatically creates their account and grants them access to every tool their role requires — instantly. The moment they are terminated in Workday, access is revoked everywhere within seconds. No manual work, no access that lingers after someone leaves."
