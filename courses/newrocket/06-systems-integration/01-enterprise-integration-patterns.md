# 01 – Enterprise Integration Patterns

As an FDE, you're the glue between ServiceNow, AI services, and the customer's existing systems. Integration is where projects succeed or fail — the AI is only as good as the data and systems it connects to.

---

## 1. Integration Architecture on ServiceNow

### The Integration Landscape

```
                    ┌──────────────────────────┐
                    │   ServiceNow Instance     │
                    │  ┌─────────────────────┐  │
                    │  │ AI Agents (Phoebe,   │  │
                    │  │ Ariel, Miles, etc.)  │  │
                    │  └─────────┬───────────┘  │
                    │            │               │
                    │  ┌─────────▼───────────┐  │
                    │  │ Integration Layer     │  │
                    │  │ ├── REST APIs         │  │
                    │  │ ├── Integration Hub   │  │
                    │  │ ├── MID Server        │  │
                    │  │ └── Import Sets       │  │
                    │  └─────────┬───────────┘  │
                    └────────────┼──────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────▼─────┐         ┌─────▼─────┐         ┌─────▼─────┐
    │  AI/LLM   │         │ Customer  │         │ SaaS      │
    │ Services  │         │ On-Prem   │         │ Services  │
    │           │         │           │         │           │
    │ OpenAI    │         │ AD/LDAP   │         │ Slack     │
    │ Azure AI  │         │ SAP       │         │ Teams     │
    │ Custom ML │         │ Oracle    │         │ Jira      │
    │ Vector DB │         │ Databases │         │ Salesforce│
    └───────────┘         └───────────┘         └───────────┘
                               │
                          ┌────▼────┐
                          │   MID   │
                          │ Server  │
                          └─────────┘
                     (bridges firewall)
```

### Integration Methods

| Method | Direction | When to Use |
|--------|-----------|------------|
| **Scripted REST API** | Inbound (external → ServiceNow) | Custom endpoints for AI services to call back |
| **REST Message (sn_ws)** | Outbound (ServiceNow → external) | Call LLM APIs, external services |
| **Integration Hub Spokes** | Outbound (no-code/low-code) | Standard integrations (Slack, Jira, cloud) |
| **MID Server** | Bidirectional (through firewall) | On-premises systems (AD, databases, monitoring) |
| **Import Sets** | Inbound (batch data) | Bulk data loads (CMDB, user sync) |
| **SOAP** | Both directions | Legacy systems (some enterprise ERPs) |
| **Email** | Both directions | Email-to-ticket, notification workflows |
| **Events/Webhooks** | Inbound | Real-time notifications from external systems |

---

## 2. REST API Integration Patterns

### Outbound: Calling External APIs

```javascript
// Pattern: Calling an LLM API with retry and error handling
var AIServiceClient = Class.create();
AIServiceClient.prototype = {
    initialize: function() {
        this.endpoint = gs.getProperty('x_myco_ai.llm_endpoint');
        this.maxRetries = 3;
    },

    call: function(prompt, context, options) {
        var attempt = 0;
        var lastError = null;

        while (attempt < this.maxRetries) {
            try {
                var request = new sn_ws.RESTMessageV2();
                request.setEndpoint(this.endpoint);
                request.setHttpMethod('POST');

                // Use Connection & Credential record (not hardcoded keys)
                request.setBasicAuth(this._getCredentialAlias());

                request.setRequestHeader('Content-Type', 'application/json');
                request.setRequestBody(JSON.stringify({
                    model: options.model || 'gpt-4',
                    messages: this._buildMessages(prompt, context),
                    temperature: options.temperature || 0.3,
                    max_tokens: options.max_tokens || 500
                }));

                // Timeout: don't hang indefinitely
                request.setHttpTimeout(30000); // 30 seconds

                var response = request.execute();
                var statusCode = response.getStatusCode();

                if (statusCode === 200) {
                    return JSON.parse(response.getBody());
                } else if (statusCode === 429) {
                    // Rate limited — wait and retry
                    attempt++;
                    gs.warn('LLM API rate limited, attempt ' + attempt);
                    // Note: can't sleep() in ServiceNow — use scheduled retry
                    lastError = 'Rate limited (429)';
                } else {
                    lastError = 'HTTP ' + statusCode + ': ' + response.getBody();
                    break; // Don't retry non-429 errors
                }
            } catch (e) {
                lastError = e.getMessage();
                attempt++;
            }
        }

        gs.error('LLM API call failed after ' + attempt + ' attempts: ' + lastError);
        return null;
    },

    type: 'AIServiceClient'
};
```

### Inbound: Receiving Webhooks

```javascript
// Scripted REST API: POST /api/x_myco_ai/webhook/ai_complete
(function process(request, response) {

    // Validate webhook signature
    var signature = request.getHeader('X-Webhook-Signature');
    if (!_validateSignature(signature, request.body.dataString)) {
        response.setStatus(401);
        response.setBody({ error: 'Invalid signature' });
        return;
    }

    var body = JSON.parse(request.body.dataString);

    // Process AI completion callback
    var gr = new GlideRecord('incident');
    if (gr.get(body.incident_sys_id)) {
        gr.work_notes = 'AI analysis complete: ' + body.summary;
        gr.setValue('x_myco_ai_confidence', body.confidence);
        gr.update();

        response.setStatus(200);
        response.setBody({ status: 'processed' });
    } else {
        response.setStatus(404);
        response.setBody({ error: 'Incident not found' });
    }

})(request, response);
```

---

## 3. MID Server: Bridging the Firewall

### What It Is

The MID (Management, Instrumentation, and Discovery) Server is a Java application that runs inside the customer's network. It acts as a proxy between ServiceNow (cloud) and on-premises systems.

```
Cloud                          Customer Network (behind firewall)
┌──────────────┐               ┌──────────────────────────┐
│  ServiceNow  │◄──── HTTPS ──►│  MID Server              │
│  Instance    │  (outbound    │  ├── Discovery probes     │
│              │   from MID)   │  ├── Orchestration tasks  │
│              │               │  └── Integration relay    │
└──────────────┘               │         │                 │
                               │    ┌────▼────┐            │
                               │    │ AD/LDAP │            │
                               │    │ SAP     │            │
                               │    │ Oracle  │            │
                               │    │ Network │            │
                               │    └─────────┘            │
                               └──────────────────────────┘
```

### When You'll Use It

- **CMDB Discovery:** Auto-discover CIs (servers, applications, network devices)
- **On-prem integrations:** Query Active Directory, internal databases, legacy APIs
- **Monitoring data:** Pull alerts from Nagios, SolarWinds, or other on-prem tools
- **Orchestration:** Execute PowerShell/SSH commands on customer servers

### MID Server for AI Workflows

```
AI Use Case: Self-healing (Miles agent)

1. Monitoring alert → ServiceNow Event Management
2. Miles correlates alert with CMDB (identifies affected CI)
3. Miles decides on remediation action (restart service)
4. ServiceNow sends orchestration task to MID Server
5. MID Server executes: SSH → server → "systemctl restart httpd"
6. MID Server reports result back to ServiceNow
7. Miles updates incident: "Service auto-restarted, monitoring for stability"
```

---

## 4. Authentication Patterns

### OAuth 2.0 (Most Common for AI Service Integration)

```
ServiceNow OAuth Flow:

1. Register OAuth Application in ServiceNow
   ├── Client ID and Secret
   ├── Token URL of the AI service
   ├── Grant type (client_credentials for service-to-service)
   └── Scopes

2. Create Connection & Credential record
   ├── Points to OAuth Application
   ├── Stores encrypted credentials
   └── Auto-refreshes tokens

3. Use in code:
   var request = new sn_ws.RESTMessageV2();
   // Credential alias handles auth automatically
   request.setAuthenticationProfile('oauth2', 'AI_Service_Profile');
```

### SSO/SAML (User-Facing Authentication)

```
Enterprise SSO Flow:

User → ServiceNow Portal → SAML Redirect → Customer's IdP (Okta, Azure AD)
                                                    │
                                                    ▼
                                            User authenticates
                                                    │
                                                    ▼
                                        SAML assertion → ServiceNow
                                                    │
                                                    ▼
                                    User logged in with enterprise credentials
```

### API Key Management

```javascript
// NEVER do this:
var apiKey = 'sk-abc123...';  // Hardcoded = security vulnerability

// DO this:
// Option 1: System Properties (encrypted)
var apiKey = gs.getProperty('x_myco_ai.api_key');

// Option 2: Connection & Credential record (preferred)
// Managed through ServiceNow's credential store
// Supports rotation, auditing, and access control

// Option 3: Mid Server credential (for on-prem)
// Stored on the MID Server, never leaves the customer's network
```

---

## 5. Cloud Platform Integration

### AWS Integration

| Service | ServiceNow Integration | AI Relevance |
|---------|----------------------|--------------|
| **S3** | Store/retrieve files via REST | Document ingestion for RAG |
| **Bedrock** | LLM API calls | Alternative to Now LLM |
| **SageMaker** | Custom model endpoints | Customer's own ML models |
| **CloudWatch** | Event ingestion via MID Server | Monitoring data for Miles |
| **Lambda** | Webhook receiver | Async processing pipeline |
| **EventBridge** | Event routing | Cross-service orchestration |

### Azure Integration

| Service | ServiceNow Integration | AI Relevance |
|---------|----------------------|--------------|
| **Azure OpenAI** | Now Assist Skill Kit (NASK) | Primary LLM provider for many customers |
| **Azure AD** | SSO/SAML, user provisioning | Identity for AI agent permissions |
| **Azure Blob** | Document storage | Document ingestion for RAG |
| **Azure Cognitive Services** | REST API | OCR, translation, speech |
| **Azure Monitor** | Event ingestion | Monitoring data for Miles |

### GCP Integration

| Service | ServiceNow Integration | AI Relevance |
|---------|----------------------|--------------|
| **Vertex AI** | REST API | LLM and custom model hosting |
| **Cloud Storage** | REST API | Document storage for RAG |
| **BigQuery** | JDBC via MID Server | Analytics data source |
| **Pub/Sub** | Webhook/REST | Event-driven integration |

---

## 6. Data Synchronization Patterns

### Import Sets (Batch)

For bulk data loads — user sync, CMDB import, historical data migration:

```
External System → CSV/JSON/XML → Import Set Table → Transform Map → Target Table

Example: Import users from HRIS
1. Scheduled job pulls CSV from SFTP (nightly)
2. Data loaded into import set table (staging)
3. Transform map matches columns to sys_user fields
4. Coalesce rules prevent duplicates (match on email)
5. New users created, existing users updated
```

### Real-Time Sync

For data that must be current:

```
Pattern: Event-Driven Sync

External System                     ServiceNow
     │                                  │
     │──── Webhook (record changed) ───►│
     │                                  ├── Scripted REST API receives
     │                                  ├── Validates and transforms
     │                                  └── Updates ServiceNow record
     │                                  │
     │◄── Outbound REST (if bidirectional)
     │                                  │
```

### CMDB Sync (Critical for AI)

The CMDB must be accurate for AI agents to work. Common sync sources:

| Source | What It Provides | Sync Method |
|--------|-----------------|-------------|
| Cloud provider APIs | VMs, containers, services | ServiceNow Discovery + Cloud Management |
| VMware vCenter | Virtual infrastructure | MID Server discovery |
| Active Directory | Users, groups, computers | MID Server + LDAP integration |
| Network monitoring | Network devices, topology | SNMP discovery via MID Server |
| ITSM history | Incident/change/problem data | Already in ServiceNow |
| Custom CMDB sources | Customer-specific CIs | Import sets + custom scripts |

---

## 7. Integration Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|-------------|-------------|-----------------|
| **Direct database access** | Bypasses ServiceNow security/audit | Use GlideRecord or REST API |
| **Synchronous external calls in before Business Rules** | Blocks form submission, terrible UX | Use async Business Rules or Flow Designer |
| **Polling instead of webhooks** | Wastes resources, introduces latency | Set up webhooks where possible |
| **No error handling on API calls** | Silent failures, missing data | Try/catch, retry logic, alerting |
| **Hardcoded credentials** | Security vulnerability, hard to rotate | Use Connection & Credential records |
| **No rate limiting** | Can overwhelm external APIs | Implement throttling in Script Includes |
| **Ignoring MID Server for on-prem** | Direct calls fail through firewalls | Deploy and configure MID Server |
| **No data validation** | Garbage in, garbage out | Validate all external data before writing to ServiceNow |
