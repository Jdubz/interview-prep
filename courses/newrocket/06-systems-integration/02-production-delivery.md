# 02 – Production Delivery

Getting from prototype to production in an enterprise environment. Security, scalability, monitoring, troubleshooting, and deployment — the concerns that separate a demo from a real system.

---

## 1. Security and Compliance

### ServiceNow Security Model

```
Access Control Stack:
├── Role-Based Access Control (RBAC)
│   ├── Roles assigned to users/groups
│   ├── Roles grant access to modules, tables, features
│   └── Example: itil role grants access to ITSM modules
│
├── Access Control Lists (ACLs)
│   ├── Row-level: who can read/write which records
│   ├── Field-level: who can see/edit which fields
│   ├── ACL rules evaluated in order (first match wins)
│   └── Example: Only assigned_to user can see work_notes
│
├── Application Scope Isolation
│   ├── Scoped apps can only access their own tables by default
│   ├── Cross-scope access must be explicitly granted
│   └── Prevents AI app from accessing unrelated data
│
└── Data Encryption
    ├── At rest: AES-256 (ServiceNow manages)
    ├── In transit: TLS 1.2+
    ├── Column-level encryption for sensitive fields
    └── Edge encryption for customer-controlled keys
```

### AI-Specific Security Concerns

| Concern | Risk | Mitigation |
|---------|------|-----------|
| **Prompt injection** | User manipulates AI through crafted input | Input sanitization, system prompt hardening, output filtering |
| **Data leakage in RAG** | AI retrieves data user shouldn't see | Enforce ACLs in retrieval pipeline, filter by user's access level |
| **PII in AI responses** | AI surfaces SSN, passwords, etc. | PII detection and redaction in AI Traffic Controller |
| **Credential exposure** | API keys in logs or responses | Use credential store, never log request bodies with auth headers |
| **Cross-tenant data** | Multi-tenant customers see each other's data | Strict tenant filtering in RAG queries and agent context |
| **Excessive AI autonomy** | Agent takes actions it shouldn't | Guardrails: confidence thresholds, human-in-the-loop for high-impact actions |

### Compliance Frameworks

Enterprise customers operate under regulatory requirements:

| Framework | Relevance | What It Means for AI |
|-----------|----------|---------------------|
| **SOC 2** | ServiceNow is SOC 2 certified | Your scoped apps inherit platform controls, but you must follow them |
| **HIPAA** | Healthcare customers | AI cannot store or transmit PHI outside approved systems |
| **GDPR** | European customers | Right to erasure applies to AI training data; data residency matters |
| **FedRAMP** | US government | ServiceNow has FedRAMP instances; restrict AI to approved models |
| **PCI DSS** | Financial services | AI cannot process or display cardholder data |

### Security Checklist for AI Deployments

```
Before go-live:
□ ACLs configured for all custom tables
□ AI agent permissions scoped to minimum necessary data
□ API credentials stored in Connection & Credential records
□ PII filtering enabled in AI responses
□ Audit logging configured for all AI decisions
□ Cross-scope access reviewed and approved
□ Input validation on all Scripted REST APIs
□ Rate limiting configured for external API calls
□ Penetration test completed (if customer requires)
□ Data retention policy documented
```

---

## 2. Scalability Patterns

### ServiceNow Performance Considerations

ServiceNow is SaaS — you don't manage infrastructure. But you can still write slow code:

| Bottleneck | Impact | Solution |
|-----------|--------|----------|
| **GlideRecord in loops** | N+1 queries, slow transactions | Batch queries, use GlideAggregate |
| **Synchronous API calls** | Block UI, timeout risk | Async Business Rules, Flow Designer |
| **Large result sets** | Memory pressure, slow rendering | Pagination (setLimit + setOffset), windowing |
| **Un-indexed queries** | Full table scans | Add database indexes on filtered fields |
| **Excessive dot-walking** | Each hop is a query | Fetch needed data in a single query where possible |
| **Heavy Business Rules** | Every insert/update slowed | Move logic to async or Flow Designer |

### Scaling AI Workloads

```
Problem: 500 tickets/day, each needs AI categorization + knowledge search + response generation

Naive approach:
  Synchronous in before Business Rule → 3-5 second delay per ticket → broken UX

Better approach:
  1. Ticket created (instant)
  2. Async Business Rule triggers Flow Designer
  3. Flow Designer runs AI pipeline:
     a. Categorization (200ms) — Predictive Intelligence (on-platform, fast)
     b. Knowledge search (500ms) — AI Search (on-platform)
     c. Response generation (2-3s) — LLM API call (async)
  4. Results written back to ticket via update
  5. Agent notified when AI suggestions are ready

Total: User sees instant ticket creation, AI results appear within 5-10 seconds
```

### Token Cost Management

```
Budget: $1,000/month for AI tokens
Average: 1,500 tokens per interaction (prompt + completion)
Cost: ~$0.01 per interaction (GPT-4 pricing)
Budget allows: ~100,000 interactions/month

Optimization strategies:
├── Cache common queries (password reset → cached response, no LLM call)
├── Use smaller models for classification (GPT-3.5 or Predictive Intelligence)
├── Reserve GPT-4/Now LLM for generation tasks
├── Implement prompt compression (shorter system prompts, fewer examples)
├── Set max_tokens limits per use case
└── Monitor via Value Realization Dashboard
```

---

## 3. Monitoring and Observability

### What to Monitor

```
Application Health:
├── Flow execution success/failure rates
├── API call latency and error rates (outbound)
├── Scripted REST API response times (inbound)
├── Business Rule execution times
└── Background job completion rates

AI Performance:
├── AI Search relevancy scores (average, distribution)
├── LLM response latency (p50, p95, p99)
├── Auto-resolution rate (by ticket type)
├── Escalation rate (AI → human)
├── Confidence score distribution
├── User satisfaction (CSAT) on AI interactions
└── Token consumption trends

Business Metrics:
├── Mean Time to Resolution (MTTR) — before vs. after AI
├── Ticket deflection rate
├── Self-service adoption
├── SLA compliance rate
├── Cost per ticket
└── FTE capacity freed
```

### ServiceNow Monitoring Tools

| Tool | What It Monitors |
|------|-----------------|
| **System Logs** | gs.info/warn/error output, script errors |
| **Flow Designer Execution History** | Flow success/failure, execution times |
| **Performance Analytics** | Business metrics, trend analysis |
| **AI Control Tower** | AI-specific decisions, governance metrics |
| **Value Realization Dashboard** | ROI metrics (newRocket-specific) |
| **Transaction Logs** | HTTP request/response details |
| **Slow Query Log** | Database queries exceeding thresholds |

### Alerting Setup

```javascript
// Example: Alert on AI service degradation
// Scheduled Job: runs every 15 minutes

(function execute() {
    var ga = new GlideAggregate('x_myco_ai_interaction');
    ga.addQuery('sys_created_on', '>=', gs.minutesAgo(15));
    ga.addQuery('status', 'error');
    ga.addAggregate('COUNT');
    ga.query();

    var errorCount = 0;
    if (ga.next()) {
        errorCount = parseInt(ga.getAggregate('COUNT'));
    }

    if (errorCount > 10) {
        gs.eventQueue('x_myco_ai.service_degradation', null,
            'AI service errors in last 15 min: ' + errorCount,
            'Check LLM API connectivity and rate limits');
    }
})();
```

---

## 4. Troubleshooting in Enterprise Environments

### Common Issues and Resolution

| Issue | Symptoms | Diagnosis | Fix |
|-------|----------|-----------|-----|
| **LLM API timeout** | AI suggestions not appearing | Check System Logs for timeout errors | Increase timeout, add retry, switch to async |
| **Poor RAG quality** | Irrelevant suggestions, low confidence | Review AI Search relevancy scores, test queries manually | Improve knowledge base, tune chunking, adjust reranker |
| **Authentication failure** | 401/403 errors in logs | Check OAuth token expiry, credential store | Refresh tokens, update credentials |
| **MID Server down** | On-prem integrations failing | MID Server status in ServiceNow, check agent process | Restart MID Server, check network |
| **Performance degradation** | Slow form loads, timeouts | Slow Query Log, Transaction Log | Optimize queries, add indexes, move to async |
| **ACL blocking AI** | Agent can't access needed data | Run ACL debug for agent user | Update ACLs for scoped app role |
| **Cross-scope access denied** | Script Include calls failing | Check application cross-scope access settings | Grant access in Application Registry |

### Debugging Techniques

```javascript
// 1. Enable session debugging
gs.setProperty('glide.security.debug.role', 'true'); // temporarily

// 2. Script debugging with detailed logging
gs.info('[AI-DEBUG] Input: ' + JSON.stringify({
    query: query,
    table: table,
    user: gs.getUserName()
}));

// 3. REST API debugging
var request = new sn_ws.RESTMessageV2();
// ... configure ...
var response = request.executeAsync(); // async version
var body = response.getBody();
gs.info('[AI-DEBUG] API response: ' + response.getStatusCode() +
        ' body length: ' + body.length);

// 4. Flow Designer: check execution history
// Navigate to: Flow Designer → Execution History → filter by flow name
// Shows step-by-step execution with inputs/outputs

// 5. ACL debugging
// Navigate to: System Diagnostics → Session Debug → Security
// Shows which ACLs were evaluated and their results
```

---

## 5. Deployment Strategies

### The Instance Promotion Pipeline

```
Development (dev)
  │
  │ Export Update Set / Source Control commit
  │
  ▼
Testing (test)
  │
  │ UAT with customer stakeholders
  │ Automated test execution (ATF)
  │
  ▼
Pre-Production (stage) — optional
  │
  │ Performance testing
  │ Security scan
  │
  ▼
Production (prod)
  │
  │ Gradual rollout
  │ Monitor for issues
  │
  ▼
Stable
```

### Gradual Rollout Strategy for AI

Don't launch AI to all users at once:

```
Phase 1: Internal pilot (1 week)
├── newRocket team + customer's ServiceNow admins
├── Test with real tickets (but don't auto-resolve)
├── Validate AI suggestions against human decisions
└── Tune configuration based on results

Phase 2: Department pilot (2 weeks)
├── One department (e.g., IT help desk)
├── AI assists human agents (suggestions, not auto-action)
├── Measure deflection rate, MTTR, satisfaction
└── Adjust confidence thresholds and guardrails

Phase 3: Full rollout (ongoing)
├── All targeted departments
├── Enable auto-resolution for high-confidence cases
├── Full Value Realization Dashboard active
└── Weekly optimization reviews for first month
```

### Rollback Plan

```
If something goes wrong:
├── Level 1: Reduce AI autonomy (disable auto-resolve, suggest only)
├── Level 2: Disable specific agent topics (e.g., turn off password reset)
├── Level 3: Disable entire AI agent (kill switch in AI Traffic Controller)
├── Level 4: Revert update set (restore previous configuration)
└── Level 5: Restore from backup (ServiceNow manages, but request from support)

Key: Always have a way to go back. Never deploy AI without a kill switch.
```

---

## 6. Handoff and Documentation

### What You Leave Behind

```
Documentation Package:
├── Architecture Document
│   ├── Solution overview diagram
│   ├── Integration map (all connected systems)
│   ├── Data flow diagram
│   ├── Security architecture
│   └── AI model and configuration details
│
├── Runbook
│   ├── Common issues and resolutions
│   ├── Monitoring dashboards and what to watch
│   ├── Escalation procedures
│   ├── Credential rotation procedures
│   └── AI model update/retraining process
│
├── Configuration Guide
│   ├── Agent configuration parameters
│   ├── Guardrail rules and how to modify
│   ├── Knowledge base maintenance procedures
│   ├── Flow Designer flow descriptions
│   └── Custom Script Include documentation
│
└── Training Materials
    ├── Admin training (how to manage the AI system)
    ├── Agent training (how to work alongside AI)
    ├── End-user training (how to interact with AI)
    └── Troubleshooting guide (first-responder level)
```

### The Handoff Meeting

```
Handoff Agenda (2 hours):
1. Solution walkthrough (30 min)
   - Demo the complete solution
   - Walk through architecture

2. Administration training (30 min)
   - How to modify agent configuration
   - How to update knowledge base
   - How to review AI performance dashboards

3. Troubleshooting workshop (30 min)
   - Common issues and how to diagnose
   - Where to find logs and metrics
   - When to escalate to newRocket support

4. Q&A and next steps (30 min)
   - Open questions
   - Phase 2 recommendations
   - Support model and contact information
```
