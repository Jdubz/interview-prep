# 01 – ServiceNow Platform Architecture and Concepts

ServiceNow is the platform newRocket builds everything on. You don't need to be a ServiceNow expert to get hired, but understanding the architecture shows you've done your homework and lets you have informed technical conversations.

> **Mental model shift:** If you come from building web apps, ServiceNow is more like Salesforce than it is like Django or Express. It's a **low-code/pro-code enterprise platform** with a database, application server, UI framework, and workflow engine all bundled together. Your JavaScript and integration skills transfer directly.

---

## 1. The Now Platform Architecture

### High-Level Stack

```
┌─────────────────────────────────────────────────┐
│  Browser / Mobile / Portal / Service Catalog     │  ← User interfaces
├─────────────────────────────────────────────────┤
│  UI Builder (Seismic) / Service Portal (Angular) │  ← Frontend frameworks
├─────────────────────────────────────────────────┤
│  Application Layer (JavaScript/Glide)            │  ← Business logic
│  ├── Business Rules                              │
│  ├── Script Includes                             │
│  ├── Flow Designer                               │
│  ├── Scheduled Jobs                              │
│  └── REST/SOAP APIs                              │
├─────────────────────────────────────────────────┤
│  Platform Services                               │  ← Core services
│  ├── Now Assist (AI)                             │
│  ├── Predictive Intelligence                     │
│  ├── Integration Hub                             │
│  ├── Performance Analytics                       │
│  └── Notifications / Events                      │
├─────────────────────────────────────────────────┤
│  Database (MariaDB)                              │  ← All data in tables
│  ├── CMDB (Configuration Management Database)    │
│  ├── Task tables (incidents, changes, requests)  │
│  └── Custom application tables                   │
├─────────────────────────────────────────────────┤
│  Infrastructure (SaaS, hosted by ServiceNow)     │  ← You don't manage this
└─────────────────────────────────────────────────┘
```

### Key Architectural Facts

- **SaaS-only** — ServiceNow hosts everything. No self-hosting. Customers get an "instance" (e.g., `acme.service-now.com`).
- **Application nodes run on Apache Tomcat** — Java under the hood, but you write JavaScript.
- **Everything is a table** — Incidents, users, servers, knowledge articles, AI models — all stored as records in tables. Think of it as a relational database you interact with through a UI and APIs, not SQL directly.
- **Two upgrade cycles per year** — Named after cities (Washington DC, Xanadu, Yokohama). Each release adds features. Customers choose when to upgrade.

---

## 2. The Data Model: Tables and Records

### Tables = Database Tables with UI

Every "thing" in ServiceNow is a record in a table. Tables have:
- **Fields** (columns) — string, integer, reference, journal, etc.
- **Records** (rows) — each with a unique `sys_id` (32-character GUID)
- **A built-in form** — auto-generated UI for creating/editing records
- **A built-in list view** — auto-generated table view with filtering

```
Table: incident
├── sys_id:        abc123def456...
├── number:        INC0012345
├── short_description: "Email not working"
├── state:         2 (In Progress)
├── assigned_to:   → sys_user record (reference field)
├── cmdb_ci:       → cmdb_ci record (what's affected)
└── priority:      2 (High)
```

### Table Inheritance

ServiceNow uses **table-per-hierarchy inheritance**. The base table is `task`. Everything that involves work extends it:

```
task (base)
├── incident       (IT issues)
├── change_request (planned changes)
├── problem        (root cause investigation)
├── sc_request     (service catalog requests)
├── sc_req_item    (individual catalog items)
├── hr_case        (HR cases)
└── [custom]       (your app's task types)
```

**Why this matters:** If you query the `task` table, you get incidents AND changes AND problems. If you query `incident`, you only get incidents. The inheritance model is how ServiceNow achieves cross-module reporting and workflow.

### Reference Fields (Foreign Keys)

Fields can reference other tables. The `assigned_to` field on an incident is a reference to the `sys_user` table. Dot-walking lets you traverse these:

```javascript
// Dot-walking: traverse references without JOINs
var assigneeName = gr.assigned_to.name;          // one hop
var managerEmail = gr.assigned_to.manager.email;  // two hops
```

---

## 3. CMDB: The Configuration Management Database

The CMDB is ServiceNow's model of your IT infrastructure — every server, application, network device, and service is a **Configuration Item (CI)**.

### Why It Matters for AI

The CMDB is the **knowledge graph of the enterprise**. When an AI agent needs to understand:
- "What servers does this application run on?"
- "Who owns this service?"
- "What was the last change made to this CI?"

...it queries the CMDB. As an FDE building AI workflows, you'll frequently connect agents to CMDB data.

### Key CMDB Tables

| Table | What It Stores |
|-------|---------------|
| `cmdb_ci` | Base CI table (all config items inherit from this) |
| `cmdb_ci_server` | Physical and virtual servers |
| `cmdb_ci_appl` | Applications |
| `cmdb_ci_service` | Business services |
| `cmdb_rel_ci` | Relationships between CIs |
| `cmdb_ci_cloud_service_account` | Cloud accounts (AWS, Azure, GCP) |

### CI Relationships

CIs are connected by relationships: "runs on," "depends on," "hosted on," "managed by." This creates a dependency graph that's critical for impact analysis.

```
Business Service: "Email"
├── Depends on → Application: "Exchange Online"
│   ├── Runs on → Server: "mail-prod-01"
│   └── Runs on → Server: "mail-prod-02"
├── Depends on → Application: "Active Directory"
│   └── Runs on → Server: "dc-prod-01"
└── Managed by → Group: "Messaging Team"
```

---

## 4. Application Scoping

### Global vs. Scoped Applications

ServiceNow has two execution contexts:

| Aspect | Global Scope | Scoped Application |
|--------|-------------|-------------------|
| **Access** | Can read/write any table | Restricted to own tables + explicitly granted access |
| **Namespace** | None — all scripts share global namespace | Isolated namespace (e.g., `x_myco_myapp`) |
| **Deployment** | Instance-level | Installable, portable, publishable to ServiceNow Store |
| **Best for** | Platform-wide customizations | Discrete applications, integrations, AI solutions |

**For newRocket AI work:** You'll almost always build **scoped applications**. newRocket's Agent Packs and Intelligence Platform are scoped apps that get installed into customer instances.

### Update Sets

ServiceNow's version control for configuration. When you change a Business Rule, form layout, or script, the change is captured in an update set. Update sets can be moved between instances (dev → test → prod).

```
Development Instance → (export update set) → Test Instance → (promote) → Production
```

Think of it as git for configuration, but clunkier. Modern approaches use **Source Control Integration** to sync with actual git repos.

---

## 5. Instance Types

Every customer has multiple instances:

| Instance | Purpose |
|----------|---------|
| **Development (dev)** | Where you build and test |
| **Test/QA** | Where UAT happens |
| **Production (prod)** | Live, customer-facing |
| **Sub-production** | For specific testing (performance, security) |
| **Personal Developer Instance (PDI)** | Free instances for learning and certification — ServiceNow gives these out |

**As an FDE**, you'll typically work in the customer's dev instance, promote to test, and support production deployments.

---

## 6. Key Terminology Cheat Sheet

| Term | What It Means | Analogy |
|------|--------------|---------|
| **Instance** | A customer's ServiceNow environment | A Heroku app / AWS account |
| **Table** | Database table with built-in UI | PostgreSQL table + auto-generated admin panel |
| **Record** | A row in a table | A database row |
| **sys_id** | 32-char GUID, primary key for everything | UUID |
| **Dot-walking** | Traversing reference fields | SQL JOIN, but in JavaScript |
| **Business Rule** | Server-side script triggered on table operations | Database trigger / middleware |
| **Client Script** | Browser-side JavaScript on forms | Frontend event handler |
| **Script Include** | Reusable server-side class/function | A module/library |
| **Flow Designer** | Visual workflow builder | Zapier / n8n / AWS Step Functions |
| **Integration Hub** | Pre-built connectors to external systems | iPaaS / middleware |
| **Update Set** | Container for configuration changes | Git branch (loosely) |
| **Scoped App** | Isolated application with its own namespace | npm package / Docker container |
| **ACL** | Access Control List — row/field-level security | IAM policy |
| **CMDB** | Configuration Management Database | Infrastructure knowledge graph |
| **CI** | Configuration Item — a thing in the CMDB | A node in the graph |
| **Catalog Item** | Something users can request | A product in a shopping cart |
| **Knowledge Article** | Documentation stored in ServiceNow | A wiki page |
