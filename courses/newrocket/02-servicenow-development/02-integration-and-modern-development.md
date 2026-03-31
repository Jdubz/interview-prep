# 02 – Integration and Modern ServiceNow Development

How to build integrations, modern UIs, and production-ready applications on ServiceNow. This is where your full-stack skills combine with platform knowledge.

---

## 1. Scripted REST APIs

ServiceNow lets you create custom REST endpoints. As an FDE building AI integrations, you'll create these to:
- Expose ServiceNow data to external AI services
- Receive webhooks and callbacks from LLM APIs
- Build custom APIs for frontend applications

### Creating a Scripted REST API

```javascript
// Resource: GET /api/x_myco_ai/incidents/active
// Script:
(function process(request, response) {

    var limit = request.queryParams.limit || '20';
    var category = request.queryParams.category;

    var gr = new GlideRecord('incident');
    gr.addQuery('state', 'IN', '1,2,3'); // New, In Progress, On Hold
    if (category) {
        gr.addQuery('category', category);
    }
    gr.setLimit(parseInt(limit));
    gr.orderByDesc('sys_created_on');
    gr.query();

    var incidents = [];
    while (gr.next()) {
        incidents.push({
            sys_id: gr.sys_id.toString(),
            number: gr.number.toString(),
            short_description: gr.short_description.toString(),
            priority: gr.priority.toString(),
            category: gr.category.toString(),
            assigned_to: gr.assigned_to.getDisplayValue(),
            created: gr.sys_created_on.toString()
        });
    }

    response.setStatus(200);
    response.setBody({
        result: incidents,
        count: incidents.length
    });

})(request, response);
```

### Authentication for REST APIs

| Method | When to Use |
|--------|------------|
| **Basic Auth** | Simple integrations, internal tools |
| **OAuth 2.0** | External AI services, production integrations |
| **API Key** | Service-to-service, middleware |
| **Mutual TLS** | High-security enterprise requirements |

```javascript
// Setting up OAuth 2.0 for external API calls
var oauth = new sn_auth.GlideOAuthClient();
var tokenResponse = oauth.requestTokenByAdminCredentials(
    'my_oauth_provider',  // OAuth Application Registry name
    'client_credentials'  // grant type
);
var token = tokenResponse.getToken().getAccessToken();
```

---

## 2. Integration Hub and Spokes

Integration Hub is ServiceNow's iPaaS — pre-built connectors (called "Spokes") that you wire into Flow Designer.

### Architecture

```
Flow Designer
  └── Action: "Send Slack Message"
      └── Spoke: Slack
          └── Connection: OAuth to customer's Slack workspace
              └── API: chat.postMessage

Flow Designer
  └── Action: "Call REST API"
      └── Spoke: REST
          └── Connection: API key + base URL
              └── API: POST /v1/chat/completions
```

### Custom Spoke Development

When no pre-built spoke exists (common for AI services):

```javascript
// Custom spoke action: Call AI Service
// Input: prompt (string), context (string)
// Output: response (string), confidence (number)

(function execute(inputs, outputs) {
    var r = new sn_ws.RESTMessageV2();
    r.setEndpoint(inputs.endpoint);
    r.setHttpMethod('POST');
    r.setRequestHeader('Content-Type', 'application/json');
    r.setRequestHeader('Authorization', 'Bearer ' + inputs.api_key);
    r.setRequestBody(JSON.stringify({
        prompt: inputs.prompt,
        context: inputs.context
    }));

    var resp = r.execute();
    var body = JSON.parse(resp.getBody());

    outputs.response = body.response;
    outputs.confidence = body.confidence;
})(inputs, outputs);
```

### Common Integration Patterns

| Pattern | Description | Use Case |
|---------|------------|----------|
| **Inbound REST** | External system calls ServiceNow API | Monitoring tool creates incident |
| **Outbound REST** | ServiceNow calls external API | Call LLM for categorization |
| **Webhook** | ServiceNow receives async callback | AI service completes analysis, notifies ServiceNow |
| **MID Server** | Proxy for on-premises systems | Access customer's internal databases/APIs |
| **Event-driven** | ServiceNow event triggers integration | Record change triggers Slack notification |
| **Batch/scheduled** | Scheduled data sync | Nightly sync of CMDB data to analytics platform |

---

## 3. Flow Designer Deep Dive

### Flow Components

```
FLOW
├── Trigger
│   ├── Record-based (created, updated, deleted)
│   ├── Schedule-based (daily, weekly, cron)
│   ├── Application-based (inbound email, REST)
│   └── Flow Logic (called by another flow)
│
├── Actions (sequential steps)
│   ├── Core: Create/Update/Delete Record, Log
│   ├── Integration Hub: REST, Email, Slack, etc.
│   ├── AI: Now Assist skills, custom AI actions
│   └── Script: Custom server-side JavaScript
│
├── Flow Logic
│   ├── If/Else (conditional branching)
│   ├── For Each (loop over records)
│   ├── Do Until (loop with condition)
│   ├── Wait For Condition (pause until condition met)
│   └── Parallel (run branches concurrently)
│
└── Error Handling
    ├── Try/Catch blocks
    └── Rollback on failure
```

### Example: AI-Powered Incident Resolution Flow

```
Trigger: Incident created (table: incident)
Condition: current.category IS NOT EMPTY

→ Action 1: AI Search
   Input: current.short_description
   Output: knowledge_articles[]

→ Decision: knowledge_articles.length > 0 AND top_score > 0.85?

   → YES branch:
     → Action 2: Call LLM (custom spoke)
        Input: incident description + top knowledge article
        Output: suggested_resolution

     → Action 3: Update Incident
        Set: work_notes = "AI-suggested resolution: " + suggested_resolution
        Set: state = 6 (Resolved) if auto_resolve_enabled

     → Action 4: Notify User
        Send email with resolution

   → NO branch:
     → Action 2: Assign to Group
        Route based on category → assignment group mapping

     → Action 3: Log
        "No confident AI match for: " + current.number
```

### Subflows (Reusable Components)

```
Subflow: "Enrich with AI Context"
  Inputs: record_sys_id, table_name

  → Get record
  → Query CMDB for related CIs
  → Search knowledge base
  → Call LLM for summary

  Outputs: ai_summary, related_cis[], knowledge_articles[]

// Reused across incident, change, and problem flows
```

---

## 4. UI Builder and Seismic Framework

### UI Builder (UIB)

The modern way to build ServiceNow UIs. Generates JSON configuration that the Seismic framework renders.

- **Workspace** — the full-page app experience for agents
- **Portal** — self-service experience for end users
- **Components** — reusable UI elements (cards, lists, forms, charts)
- **Data resources** — server-side data fetching for components

### Seismic Framework

ServiceNow's custom Web Components framework:
- **Shadow DOM** for component isolation
- **GraphQL** as the primary communication protocol between UI and server
- Replaces the legacy AngularJS-based Service Portal

### When You'll Touch UI

As an FDE, you'll primarily build backend/integration work, but you may:
- Configure Agent Workspace views for AI-assisted workflows
- Build portal pages that surface AI agent interactions
- Create dashboards showing AI performance metrics (Value Realization Dashboard)

---

## 5. Scoped Application Development

### Application Structure

```
Scoped App: x_myco_ai_assistant
├── Tables
│   ├── x_myco_ai_conversation      (chat sessions)
│   ├── x_myco_ai_interaction       (individual messages)
│   └── x_myco_ai_feedback          (user ratings)
├── Script Includes
│   ├── AIServiceHelper              (LLM API wrapper)
│   ├── KnowledgeSearcher            (RAG retrieval)
│   └── ConversationManager          (session management)
├── Business Rules
│   ├── Auto-categorize on insert    (incident table)
│   └── Feedback trigger             (feedback table)
├── Scripted REST APIs
│   ├── /chat                        (conversation endpoint)
│   └── /feedback                    (rating endpoint)
├── Flows
│   ├── AI Incident Triage           (main workflow)
│   └── Escalation Handler           (fallback)
├── UI Components
│   ├── AI Chat Widget               (portal component)
│   └── Resolution Confidence Badge  (workspace component)
├── Roles
│   ├── x_myco_ai_assistant.admin
│   └── x_myco_ai_assistant.user
└── Properties
    ├── x_myco_ai.llm_endpoint
    ├── x_myco_ai.api_key
    ├── x_myco_ai.auto_resolve_threshold
    └── x_myco_ai.max_tokens
```

### Cross-Scope Access

Scoped apps can't access other scopes by default. You explicitly grant access:

```javascript
// In your scoped app, to read incident data:
// 1. Set "Accessible from: All application scopes" on the incident table (admin does this)
// 2. Or use the Table API with elevated privileges

// Runtime access check
if (gs.hasRole('x_myco_ai_assistant.admin')) {
    // Can configure AI settings
}
```

---

## 6. Development Best Practices for Enterprise

### Security

- **Never hardcode credentials** — use System Properties or Connection & Credential records
- **Always validate input** — especially on Scripted REST APIs (injection attacks apply here too)
- **Use ACLs** — control who can read/write your tables at the row and field level
- **Encrypt sensitive data** — use `GlideEncrypter` for storing tokens, keys
- **Audit trail** — ServiceNow auto-logs all changes, but add custom audit for AI decisions

### Performance

- **Index your tables** — add indexes for fields used in queries
- **Use GlideAggregate** for counts — never iterate records just to count
- **Async processing** — use async Business Rules or scheduled jobs for heavy operations
- **Cache API responses** — ServiceNow has a caching framework; use it for repeated LLM calls
- **Pagination** — always paginate API responses (use `sysparm_limit` and `sysparm_offset`)

### Deployment Pipeline

```
Developer's PDI (Personal Developer Instance)
  → Team Dev Instance (shared development)
    → Test/QA Instance (UAT with customer)
      → Production Instance

Changes tracked via:
  - Update Sets (traditional)
  - Source Control Integration (git-based, modern)
  - App Engine Studio (low-code deployment)
```

### Testing

```javascript
// ServiceNow ATF (Automated Test Framework)
// Create test suites that run in the platform

// Example: Test that AI categorization works
// Step 1: Create incident with known description
// Step 2: Assert category field was set correctly
// Step 3: Assert work_notes contain "Auto-categorized"
// Step 4: Assert confidence threshold was met
```
