# 01 – ServiceNow Scripting and Glide APIs

ServiceNow scripting is JavaScript — but within a specific framework. If you know JS/TS, you already know 80% of what you need. This module covers the ServiceNow-specific APIs and patterns that make up the other 20%.

> **Key fact:** ServiceNow uses ECMAScript 2021 (ES12) as of the Washington DC release. Arrow functions, optional chaining, destructuring, `let`/`const` — all supported on the server side.

---

## 1. Server-Side vs. Client-Side

ServiceNow scripting runs in two contexts:

| Aspect | Server-Side | Client-Side |
|--------|------------|-------------|
| **Runs where** | Application server (Tomcat/JVM) | User's browser |
| **Language** | JavaScript (Rhino/GraalJS engine) | JavaScript (browser engine) |
| **API access** | Full Glide API (GlideRecord, GlideSystem, etc.) | Limited API (GlideForm, GlideAjax to call server) |
| **Data access** | Direct database queries via GlideRecord | Must call server via GlideAjax or REST |
| **Script types** | Business Rules, Script Includes, Scheduled Jobs, REST APIs | Client Scripts, UI Policies, UI Actions |
| **Performance** | Runs on server — affects all users | Runs in browser — affects one user |
| **Security** | Trusted — runs in server context | Untrusted — user can inspect/modify |

**Rule of thumb:** Do as much as possible server-side. Client-side is for UI behavior only (showing/hiding fields, validation before submit, user feedback).

---

## 2. GlideRecord: The Core Data API

GlideRecord is how you query and manipulate data in ServiceNow. Think of it as an ORM that maps directly to tables.

### Basic CRUD Operations

```javascript
// CREATE
var gr = new GlideRecord('incident');
gr.initialize();
gr.short_description = 'Server unreachable';
gr.category = 'network';
gr.priority = 2;
gr.insert();
// Returns sys_id of new record

// READ (query)
var gr = new GlideRecord('incident');
gr.addQuery('priority', 1);
gr.addQuery('state', '!=', 7); // not closed
gr.orderByDesc('sys_created_on');
gr.setLimit(10);
gr.query();

while (gr.next()) {
    gs.info(gr.number + ': ' + gr.short_description);
}

// UPDATE
var gr = new GlideRecord('incident');
gr.get('sys_id_here'); // or gr.get('number', 'INC0012345')
gr.state = 6; // Resolved
gr.resolution_notes = 'Applied patch KB0012345';
gr.update();

// DELETE
var gr = new GlideRecord('incident');
gr.get('sys_id_here');
gr.deleteRecord();
```

### Query Operators

```javascript
gr.addQuery('priority', '<=', 2);              // comparison
gr.addQuery('short_description', 'CONTAINS', 'email'); // string match
gr.addQuery('assigned_to', 'IN', 'id1,id2,id3');       // in list
gr.addQuery('resolved_at', 'ISNOTEMPTY', '');           // not null
gr.addEncodedQuery('priority=1^state=2');                // encoded query string (from URL)
```

### GlideAggregate: Efficient Counting and Grouping

```javascript
// Don't do this — fetches all records just to count
var gr = new GlideRecord('incident');
gr.addQuery('priority', 1);
gr.query();
var count = gr.getRowCount(); // BAD: loads all rows

// Do this instead
var ga = new GlideAggregate('incident');
ga.addQuery('priority', 1);
ga.addAggregate('COUNT');
ga.query();
if (ga.next()) {
    var count = ga.getAggregate('COUNT'); // GOOD: database-level count
}

// Group by
var ga = new GlideAggregate('incident');
ga.addAggregate('COUNT');
ga.groupBy('category');
ga.query();
while (ga.next()) {
    gs.info(ga.category + ': ' + ga.getAggregate('COUNT'));
}
```

### Dot-Walking (Reference Traversal)

```javascript
var gr = new GlideRecord('incident');
gr.get('sys_id_here');

// Traverse references without JOINs
var assigneeName = gr.assigned_to.name;              // user's name
var assigneeEmail = gr.assigned_to.email;            // user's email
var managerName = gr.assigned_to.manager.name;       // user's manager's name
var serverName = gr.cmdb_ci.name;                    // affected CI's name

// GOTCHA: dot-walking returns a GlideElement, not a string
// Use toString() or getValue() when you need the actual value
var userId = gr.assigned_to.toString();              // sys_id as string
var userName = gr.assigned_to.getDisplayValue();     // display value
```

### Performance Patterns

```javascript
// BAD: querying inside a loop (N+1 problem)
var incidents = new GlideRecord('incident');
incidents.query();
while (incidents.next()) {
    var user = new GlideRecord('sys_user');
    user.get(incidents.assigned_to);  // query per incident!
    gs.info(user.name);
}

// GOOD: use dot-walking (single query with JOIN)
var incidents = new GlideRecord('incident');
incidents.query();
while (incidents.next()) {
    gs.info(incidents.assigned_to.name);  // dot-walk, no extra query
}

// GOOD: batch with encoded query
var userIds = [];
// ... collect IDs ...
var users = new GlideRecord('sys_user');
users.addQuery('sys_id', 'IN', userIds.join(','));
users.query();

// ALWAYS set limits when you don't need all records
gr.setLimit(100);

// ALWAYS use indexed fields in queries (check sys_dictionary for indexes)
```

---

## 3. GlideSystem (gs): System Utilities

```javascript
// Logging (appears in System Log)
gs.info('Processing incident: ' + gr.number);
gs.warn('SLA approaching breach for: ' + gr.number);
gs.error('Failed to update record: ' + gr.sys_id);

// Current user context
gs.getUserID();           // sys_id of current user
gs.getUserName();         // username
gs.getUserDisplayName();  // full name
gs.hasRole('admin');      // role check

// Date/time
gs.now();                 // current datetime
gs.nowDateTime();         // current datetime string
gs.daysAgo(7);            // datetime 7 days ago (for queries)

// Properties (system config)
gs.getProperty('glide.servlet.uri'); // instance URL

// Events
gs.eventQueue('incident.resolved', gr, gr.assigned_to, gr.number);
```

---

## 4. Business Rules

Business Rules are server-side scripts that run when records are inserted, updated, deleted, or queried. They're the most common server-side script type.

### When They Run

| Type | When | Use Case |
|------|------|----------|
| **before** | Before the database operation | Validate/transform data, set defaults |
| **after** | After the database operation | Notifications, create related records, integrations |
| **async** | After, but in a separate thread | Heavy processing, external API calls |
| **display** | When the form loads | Calculate display values (rare, prefer client-side) |

### Example: Auto-Categorize with AI

```javascript
// Business Rule: before insert on incident table
// Condition: current.category.nil()
(function executeRule(current, previous) {

    // Call a Script Include that wraps the AI categorization API
    var categorizer = new AICategorizer();
    var result = categorizer.categorize(current.short_description.toString());

    if (result.confidence > 0.85) {
        current.category = result.category;
        current.subcategory = result.subcategory;
        current.work_notes = 'Auto-categorized by AI (confidence: ' +
            Math.round(result.confidence * 100) + '%)';
    }

})(current, previous);
```

### Business Rule Best Practices

1. **Keep them short** — complex logic goes in Script Includes
2. **Use conditions** — filter in the condition field, not in script (better performance)
3. **Avoid GlideRecord queries in before rules** — they're synchronous and block the transaction
4. **Use async for external calls** — never call REST APIs in synchronous business rules
5. **`current` vs `previous`** — `current` is the new values, `previous` is the old. Use for change detection.

---

## 5. Script Includes

Reusable server-side classes and functions. The equivalent of a module/library.

```javascript
// Script Include: AICategorizer
// Accessible from: This application scope
// Client callable: false (set to true for GlideAjax)

var AICategorizer = Class.create();
AICategorizer.prototype = {
    initialize: function() {
        this.endpoint = gs.getProperty('x_myco_ai.categorizer_url');
        this.apiKey = gs.getProperty('x_myco_ai.api_key');
    },

    categorize: function(text) {
        var request = new sn_ws.RESTMessageV2();
        request.setEndpoint(this.endpoint);
        request.setHttpMethod('POST');
        request.setRequestHeader('Authorization', 'Bearer ' + this.apiKey);
        request.setRequestHeader('Content-Type', 'application/json');
        request.setRequestBody(JSON.stringify({
            text: text,
            model: 'incident-categorizer-v2'
        }));

        var response = request.execute();
        var body = JSON.parse(response.getBody());

        return {
            category: body.category,
            subcategory: body.subcategory,
            confidence: body.confidence
        };
    },

    type: 'AICategorizer'
};
```

### Client-Callable Script Includes (GlideAjax)

When client-side code needs server-side data:

```javascript
// Server-side Script Include (client_callable = true)
var MyAjaxUtil = Class.create();
MyAjaxUtil.prototype = Object.extendsObject(AbstractAjaxProcessor, {

    getAssignmentGroupMembers: function() {
        var groupId = this.getParameter('sysparm_group_id');
        var members = [];

        var gr = new GlideRecord('sys_user_grmember');
        gr.addQuery('group', groupId);
        gr.query();
        while (gr.next()) {
            members.push({
                sys_id: gr.user.toString(),
                name: gr.user.getDisplayValue()
            });
        }

        return JSON.stringify(members);
    },

    type: 'MyAjaxUtil'
});

// Client-side call
var ga = new GlideAjax('MyAjaxUtil');
ga.addParam('sysparm_name', 'getAssignmentGroupMembers');
ga.addParam('sysparm_group_id', g_form.getValue('assignment_group'));
ga.getXMLAnswer(function(answer) {
    var members = JSON.parse(answer);
    // Update form UI with members
});
```

---

## 6. Client Scripts

Browser-side JavaScript that runs on ServiceNow forms.

| Type | When It Runs |
|------|-------------|
| **onLoad** | When the form loads |
| **onChange** | When a specific field value changes |
| **onSubmit** | When the user submits the form |
| **onCellEdit** | When editing in list view |

### GlideForm (g_form) API

```javascript
// Get/set field values
g_form.getValue('priority');           // returns the value
g_form.getDisplayValue('assigned_to'); // returns display text
g_form.setValue('category', 'network');

// Show/hide fields
g_form.setVisible('subcategory', true);
g_form.setDisplay('resolution_notes', false);

// Make fields mandatory or read-only
g_form.setMandatory('resolution_notes', true);
g_form.setReadOnly('category', true);

// Field messages
g_form.showFieldMsg('email', 'This email is not in our directory', 'error');
g_form.hideFieldMsg('email');

// Add/remove options from choice fields
g_form.addOption('priority', '0', 'Emergency');
g_form.removeOption('state', '7');
```

### Example: Dynamic Form Behavior

```javascript
// onChange Client Script on 'category' field
function onChange(control, oldValue, newValue, isLoading) {
    if (isLoading) return; // don't run on form load

    // Show resolution_code only for certain categories
    var showResolution = (newValue === 'software' || newValue === 'hardware');
    g_form.setVisible('resolution_code', showResolution);

    // Auto-set assignment group based on category
    if (newValue === 'network') {
        g_form.setValue('assignment_group', 'NETWORK_TEAM_SYS_ID');
    }
}
```

---

## 7. Key Scripting Patterns for AI Integration

### Calling External LLM APIs

```javascript
// Script Include for calling an LLM API
callLLM: function(prompt, context) {
    var request = new sn_ws.RESTMessageV2();
    request.setEndpoint('https://api.openai.com/v1/chat/completions');
    request.setHttpMethod('POST');
    request.setRequestHeader('Authorization', 'Bearer ' + this.apiKey);
    request.setRequestHeader('Content-Type', 'application/json');

    var body = {
        model: 'gpt-4',
        messages: [
            { role: 'system', content: 'You are an IT support assistant.' },
            { role: 'user', content: prompt }
        ],
        temperature: 0.3,
        max_tokens: 500
    };

    if (context) {
        body.messages.splice(1, 0, {
            role: 'system',
            content: 'Context from knowledge base:\n' + context
        });
    }

    request.setRequestBody(JSON.stringify(body));
    var response = request.execute();

    if (response.getStatusCode() === 200) {
        var result = JSON.parse(response.getBody());
        return result.choices[0].message.content;
    }

    gs.error('LLM API call failed: ' + response.getStatusCode());
    return null;
},

// Using it in a Flow Designer script action
(function execute(inputs, outputs) {
    var ai = new AIServiceHelper();
    var context = ai.searchKnowledgeBase(inputs.query);
    var response = ai.callLLM(inputs.query, context);
    outputs.ai_response = response;
})(inputs, outputs);
```

### Querying CMDB for AI Context

```javascript
// Get the dependency tree for a CI (useful for impact analysis)
getImpactedServices: function(ciSysId) {
    var services = [];
    var rel = new GlideRecord('cmdb_rel_ci');
    rel.addQuery('child', ciSysId);
    rel.addQuery('type.name', 'Depends on::Used by');
    rel.query();

    while (rel.next()) {
        var parent = rel.parent.getRefRecord();
        if (parent.sys_class_name === 'cmdb_ci_service') {
            services.push({
                sys_id: parent.sys_id.toString(),
                name: parent.name.toString(),
                owned_by: parent.owned_by.getDisplayValue()
            });
        }
    }
    return services;
}
```
