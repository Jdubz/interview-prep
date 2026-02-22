# Module 08: Observability

## Overview

Observability is the ability to understand the internal state of a system by examining its external outputs. For senior engineers, observability goes beyond "add logging and dashboards" -- it means building systems where any question about behavior can be answered without deploying new code. This module covers the three pillars (logs, metrics, traces), SLI/SLO-based reliability engineering, alerting that does not cause fatigue, incident response, and the organizational practices that turn data into actionable insight.

---

## Table of Contents

1. [The Three Pillars](#the-three-pillars)
2. [Structured Logging](#structured-logging)
3. [Metrics](#metrics)
4. [Distributed Tracing](#distributed-tracing)
5. [SLIs, SLOs, and SLAs](#slis-slos-and-slas)
6. [Alerting Strategies](#alerting-strategies)
7. [Incident Response](#incident-response)
8. [Dashboards](#dashboards)
9. [Interview Questions](#interview-questions)

---

## The Three Pillars

### Logs, Metrics, Traces -- Complementary, Not Redundant

Each pillar answers a different question:

| Pillar | What It Answers | Granularity | Cost Profile |
|--------|----------------|-------------|--------------|
| **Logs** | "What happened?" | Individual events | High volume, high cost at scale |
| **Metrics** | "How is the system performing?" | Aggregated time series | Low volume, low cost |
| **Traces** | "Why is this request slow?" | Per-request path | Medium volume, medium cost |

**Why you need all three:**
- Metrics tell you something is wrong (error rate spike)
- Traces tell you where in the request path the problem is (the database call in service B)
- Logs tell you exactly what happened (the specific SQL query that failed)

```
Alert fires (metric) --> Investigate trace for a failing request --> Read logs for the specific error
```

This is the observability investigation loop. If any pillar is missing, you hit a dead end.

### The Observability Maturity Model

| Level | Capability |
|-------|-----------|
| 0 | No observability. SSH into production and tail logs. |
| 1 | Centralized logging. Basic metrics dashboards. |
| 2 | Structured logging with correlation IDs. SLOs defined. Basic tracing. |
| 3 | Full distributed tracing. SLO-based alerting. Automated incident response. |
| 4 | Observability-driven development. Production debugging without new deploys. |

---

## Structured Logging

### JSON Logs

Unstructured logs are for humans reading a terminal. Structured logs are for machines parsing at scale.

```json
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "ERROR",
  "message": "Failed to process order",
  "service": "order-service",
  "version": "2.3.1",
  "trace_id": "abc123def456",
  "span_id": "789ghi",
  "order_id": "12345",
  "user_id": "usr_789",
  "error": "insufficient_inventory",
  "sku": "WIDGET-001",
  "requested_quantity": 5,
  "available_quantity": 2
}
```

**Benefits:** Query by any field, aggregate and count, correlate across services using trace_id, build dashboards from log data, machine-parseable by default.

### Log Levels

| Level | When to Use | Example |
|-------|------------|---------|
| **DEBUG** | Diagnostic information for developers | SQL queries, cache hits/misses |
| **INFO** | Normal operational events | Request completed, job started, user logged in |
| **WARN** | Unexpected but recoverable situations | Retry attempt, deprecated API used |
| **ERROR** | Operation failed but service continues | Failed request, external service down |
| **FATAL** | Service cannot continue, shutting down | Cannot connect to database, OOM |

**Common mistake**: Logging expected outcomes at WARN or ERROR. A user entering an invalid email is INFO (normal application behavior), not ERROR (unexpected failure).

### Correlation IDs

A correlation ID ties together all log entries for a single request, across all services. Generate at the API gateway, propagate via `X-Request-ID` header, attach to async context for automatic inclusion in all logs.

```javascript
app.use((req, res, next) => {
  const correlationId = req.headers['x-request-id'] || crypto.randomUUID();
  req.correlationId = correlationId;
  res.setHeader('X-Request-ID', correlationId);
  asyncLocalStorage.run({ correlationId }, () => next());
});
```

### Log Aggregation Platforms

| Platform | Type | Strengths |
|----------|------|-----------|
| **ELK Stack** | Self-hosted / Cloud | Full-text search, powerful queries, mature |
| **Grafana Loki** | Self-hosted / Cloud | Label-indexed (not full-text), cost-efficient |
| **Datadog Logs** | SaaS | Integrated with metrics/traces, easy setup |
| **Splunk** | SaaS / Self-hosted | Enterprise, powerful SPL query language |

Loki indexes labels only (cheaper to operate). Elasticsearch indexes everything (faster full-text search).

### Avoiding Log Noise

- Do not log sensitive data (PII, tokens, credit card numbers)
- Do not log per-request at DEBUG in production -- use sampling or dynamic log levels
- Do not log successful health checks -- they dominate volume with zero signal
- Do log at service boundaries: incoming/outgoing requests, key decisions
- Use sampling for high-volume paths: 1% of successes, 100% of errors
- Set retention policies: 7 days for DEBUG, 30 days for INFO, 90+ days for ERROR

---

## Metrics

### RED Method (for Services)

| Signal | What to Measure | Prometheus Type |
|--------|----------------|----------------|
| **R**ate | Requests per second | Counter |
| **E**rrors | Failed requests per second | Counter |
| **D**uration | Request latency distribution | Histogram |

```promql
rate(http_requests_total[5m])
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

### USE Method (for Resources)

| Signal | What to Measure | Example |
|--------|----------------|---------|
| **U**tilization | % of resource capacity used | CPU usage 75% |
| **S**aturation | Work queued beyond capacity | Run queue length, swap usage |
| **E**rrors | Resource error events | Disk errors, NIC errors, OOM kills |

### Prometheus Metric Types

| Type | Description | Use Case | Example |
|------|------------|----------|---------|
| **Counter** | Monotonically increasing | Total requests, errors, bytes | `http_requests_total` |
| **Gauge** | Goes up and down | Queue depth, connections, temperature | `db_connections_active` |
| **Histogram** | Distribution in buckets | Latency, response sizes | `http_request_duration_seconds` |
| **Summary** | Client-side quantiles | Exact quantiles (prefer histogram) | `rpc_duration_seconds` |

Histogram: server-side aggregation, can combine across instances, configurable buckets -- preferred for most use cases. Summary: cannot aggregate across instances, use only when exact quantiles are critical.

### Instrumentation Example

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests',
    ['method', 'endpoint', 'status'])

REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])

@app.route('/api/orders', methods=['POST'])
def create_order():
    with REQUEST_LATENCY.labels(method='POST', endpoint='/api/orders').time():
        try:
            result = process_order()
            REQUEST_COUNT.labels(method='POST', endpoint='/api/orders', status='200').inc()
            return result
        except Exception:
            REQUEST_COUNT.labels(method='POST', endpoint='/api/orders', status='500').inc()
            raise
```

### PromQL Basics

```promql
http_requests_total{job="api-server"}                          # instant vector
rate(http_requests_total[5m])                                   # per-second rate
sum(rate(http_requests_total[5m])) by (endpoint)               # aggregation
topk(5, sum(rate(http_requests_total[5m])) by (endpoint))      # top 5
```

### Custom Metrics

Instrument business logic: orders created (by payment method), order value distribution, cart abandonment, cache hit/miss rates, queue depth. **Label cardinality warning:** Every unique label combination creates a new time series. Never use user IDs or request IDs as labels -- they will overwhelm Prometheus.

---

## Distributed Tracing

### OpenTelemetry

OTel is the industry standard for vendor-neutral instrumentation.

**Core concepts:** A Trace is the entire journey of a request across services. A Span is a single operation within a trace (name, start time, duration, attributes, events, status, child spans). Context propagation passes trace context between services via the W3C `traceparent` header.

### Auto-Instrumentation

```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://otel-collector:4318/v1/traces',
  }),
  instrumentations: [getNodeAutoInstrumentations()],
});
sdk.start();
```

Covers: HTTP clients/servers, database clients (pg, mysql2, redis), gRPC, message brokers (Kafka, RabbitMQ), AWS SDK calls.

### Manual Spans

For business logic that auto-instrumentation does not cover, create spans explicitly with attributes, status codes, and exception recording.

### Sampling Strategies

| Strategy | Description | Tradeoff |
|----------|------------|----------|
| **Head-based** | Decide at trace start (e.g., 10% random) | Simple, misses interesting traces |
| **Tail-based** | Decide after completion (keep errors, slow) | Captures all interesting, requires buffering |
| **Rate-limiting** | N traces per second | Predictable cost, may miss bursts |
| **Priority** | Always sample errors and slow requests | Best signal, more complex setup |

### Tracing Backends

**Jaeger:** Mature, K8s-native. **Grafana Tempo:** Cost-efficient (object storage). **Datadog APM:** Full-stack integration. **Honeycomb:** High-cardinality queries, BubbleUp.

---

## SLIs, SLOs, and SLAs

### Definitions

- **SLI (Service Level Indicator)**: Quantitative measure of service behavior. "What are we measuring?"
- **SLO (Service Level Objective)**: Target value for an SLI. "What is acceptable?"
- **SLA (Service Level Agreement)**: Contractual commitment with consequences. "What did we promise?"

SLO should be stricter than SLA -- missing your SLO gives you time to fix before breaching your SLA.

### Defining Good SLIs

SLIs should measure what users experience, not internal state. "CPU utilization" is not an SLI. "Successful request rate" is. Measure at the edge (load balancer, client-side) when possible.

| Service Type | SLI | How to Measure |
|-------------|-----|---------------|
| HTTP API | Availability | (non-5xx responses) / (total responses) |
| HTTP API | Latency | (requests < 500ms) / (total requests) |
| Data pipeline | Freshness | current_time - last_successful_run |
| Storage | Durability | 1 - (data_loss_events / total_objects) |

### Error Budgets

Error budget = inverse of SLO. At 99.9%: 43.2 min/month downtime allowed. At 99.95%: 21.6 min. At 99.99%: 4.32 min.

**Policy:** Budget >50%: deploy freely. Budget <25%: slow down, focus reliability. Exhausted: freeze features, all effort on reliability.

### Burn Rate Alerting

Alert based on the rate at which you consume your error budget, not on static thresholds.

```
Burn rate = (actual error rate) / (error rate allowed by SLO)
SLO 99.9%, actual error rate 1.0% -> burn rate = 10x
```

**Multi-window burn rate alerts (Google SRE):**

| Severity | Burn Rate | Long Window | Short Window | Budget Consumed |
|----------|-----------|-------------|--------------|-----------------|
| Page (critical) | 14.4x | 1 hour | 5 minutes | 2% in 1 hour |
| Page (high) | 6x | 6 hours | 30 minutes | 5% in 6 hours |
| Ticket (medium) | 3x | 1 day | 2 hours | 10% in 1 day |

Both windows must fire simultaneously -- long window catches sustained issues, short window confirms it is still happening.

---

## Alerting Strategies

### Reducing Alert Fatigue

Symptoms: acknowledging without investigating, permanent suppression, "ignore these alerts" tribal knowledge, pages during maintenance windows.

Solutions: delete alerts with no action in 30 days, use burn rate instead of static thresholds, route by severity (not everything is a page), aggregate related alerts, use maintenance windows.

### Actionable Alerts

Every alert answers: (1) What is happening? (2) Why does it matter? (3) What should I do?

```
FIRING: Order Processing Degraded
  Summary: Order service p99 latency exceeded 2s for 15 minutes
  Impact: Slow checkout. Error rate 3.2% (SLO: 0.1%). Burn rate: 32x.
  Runbook: https://wiki.internal/runbooks/order-service-latency
  Dashboard: https://grafana.internal/d/order-service
```

### Severity Levels and Escalation

| Severity | Response Time | Notification |
|----------|--------------|-------------|
| **P1 - Critical** | Immediate | Page on-call |
| **P2 - High** | < 30 min | Page during business hours, Slack otherwise |
| **P3 - Medium** | < 4 hours | Slack + ticket |
| **P4 - Low** | Next business day | Ticket only |

### Runbooks

Every paging alert links to a runbook: quick assessment steps (dashboard, recent deploys, dependent services), common causes with fixes, escalation path. Without runbooks, on-call engineers waste time rediscovering known solutions.

---

## Incident Response

### On-Call Practices

Healthy on-call: fair rotation, compensated (time off or pay), max one page per shift on average, handoff with context, authority to make operational decisions (rollback, scale up, circuit breaker).

### Incident Commander Role

For P1/P2 incidents, designate an IC who coordinates but does NOT debug. IC responsibilities: declare incident, set severity, create communication channel, assign roles, make decisions (rollback or not, involve other teams), track timeline, communicate updates every 15-30 minutes, schedule postmortem.

### Communication

**Internal:** Dedicated Slack channel per incident. Status updates every 15-30 minutes, even with no news. Pin current status, impact, and assignments.

**External (status page):** Acknowledge within 5 minutes. Update regularly. Be honest about impact. Confirm resolution with brief explanation.

### Blameless Postmortems

Focus on systemic causes, not individuals. "Human error" is never a root cause -- the system allowed the error. Ask "what made this possible?" not "who did this?"

**Template:** Summary, impact (users, duration, revenue), timeline, root cause, contributing factors, what went well, what could improve, action items with owners and due dates.

### Follow-Up Tracking

Track actions alongside regular work (Jira, Linear, Issues). Assign owners and due dates. Review in weekly meetings. Escalate overdue items. Measure: % completed within 30 days.

---

## Dashboards

### Golden Signals Dashboard

Every service: Latency (p50/p95/p99), Traffic (RPS), Errors (rate, count), Saturation (CPU, memory, connection pool). Include recent deployments for correlation.

### Service Health Dashboard

Platform overview: all services with status, SLO, error budget remaining, active incidents, deploy count.

### Business Metrics

Orders per minute (vs last week), checkout conversion, payment success rate by provider, search CTR, API usage by tier. Observability is not just for engineers.

### Avoiding Sprawl

Hierarchy (platform -> service), ownership (unowned = deleted), standardized templates, quarterly usage review, link from alerts, deployment annotations.

---

## Interview Questions

**Q: You join a team with no observability. How do you build it from scratch?**

A: Phased. Week 1-2: structured JSON logging with correlation IDs, basic RED metrics with Prometheus/Grafana, /healthz endpoints. Week 3-4: define SLIs/SLOs with product, burn-rate alerts, golden signals dashboards. Month 2: distributed tracing with OTel auto-instrumentation, incident process with postmortem template. Month 3+: custom business metrics, anomaly detection, profiling, RUM.

**Q: Your SLO is 99.95% but you are at 99.7%. What do you do?**

A: Triage errors by endpoint/region/cause. Fix top contributors (rollback, scale, circuit breaker). If budget exhausted, invoke error budget policy -- freeze features, prioritize reliability. Review 30 days of incidents for patterns. Evaluate if SLO is appropriate.

**Q: How do you handle alert fatigue (50+ pages/week)?**

A: Audit last month -- was each actionable? Delete/suppress non-actionable. Consolidate per-instance into aggregate alerts. Switch to SLO-based burn rate alerting. Fix underlying instability. Target: 0-2 pages per shift.

**Q: How would you implement tracing across 50 microservices?**

A: Incremental. Deploy OTel Collector as DaemonSet. Auto-instrumentation first (80% of value). Manual spans for top 3-5 user flows. Tail-based sampling (100% errors, 5% baseline). Verify traceparent propagation. Expand gradually.

**Q: Walk me through running a major incident as IC.**

A: (1) Declare, create channel, post summary with impact. (2) Page relevant leads, assign debug vs comms roles. (3) Parallel investigation. (4) Status page within 5 min, internal updates every 15 min. (5) If no root cause in 15 min, mitigate first (rollback, scale, fallback). (6) Confirm resolution with monitoring. (7) Postmortem within 48 hours.

---

## Key Takeaways

1. **Observability is not monitoring**: Monitoring tells you when something breaks. Observability lets you ask arbitrary questions without new deploys.
2. **Structure everything**: Structured logs, standardized metrics, correlated traces.
3. **SLOs are the foundation**: Without SLOs, every alert is an opinion. With SLOs, every alert is grounded in user impact.
4. **Burn rate alerting reduces noise dramatically** compared to static thresholds.
5. **Blameless postmortems create learning organizations**: Improve the system, not punish individuals.
6. **Invest in tracing early**: Distributed tracing is the fastest path to understanding cross-service behavior.
7. **The investigation loop: metrics alert, traces locate, logs explain.**
