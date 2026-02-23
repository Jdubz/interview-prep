# Temporal Technologies — Interview Study Guide

**Role**: Senior Software Engineer — Billing, Metering & Cloud Nexus
**Team**: Product scalability, customer experience, and revenue growth
**Stack**: Go, PostgreSQL, Redshift, Kinesis, S3, Kubernetes, Stripe, Metronome, Temporal

---

## Study Plan (Priority Order)

| # | Topic | Time | What to Study |
|---|-------|------|---------------|
| 1 | Company & Product | 2 hrs | Temporal's mission, durable execution, server architecture, Cloud vs self-hosted, values |
| 2 | Temporal SDK Patterns | 3 hrs | Workflows, activities, workers, determinism, signals/queries, sagas, versioning, continue-as-new |
| 3 | Billing & Metering Systems | 3 hrs | Usage-based billing pipeline, metering, Stripe, marketplace integrations, double-entry ledger |
| 4 | Interview Practice | 3 hrs | System design walkthroughs, behavioral stories, questions to ask |
| 5 | Go for Systems | 2 hrs | Concurrency, error handling, interfaces, context, gRPC, testing |
| 6 | Data Systems | 2 hrs | PostgreSQL at scale, Redshift, Kinesis, S3, pipeline patterns |
| 7 | Platform Architecture | 1.5 hrs | Cloud Nexus, multi-cloud, tenant isolation, control/data plane separation |

### Cross-References to Existing Courses

- `courses/golang/` — Go language fundamentals
- `courses/infrastructure/02-databases-at-scale/` — Database scaling patterns
- `courses/infrastructure/04-message-queues/` — Kafka, event sourcing, CQRS
- `courses/infrastructure/06-containers-orchestration/` — Kubernetes
- `courses/infrastructure/08-observability/` — Monitoring, tracing, alerting

---

## 1. Company & Product

**What is Temporal?** A durable execution platform — you write workflows as code, and Temporal guarantees they run to completion even through failures, restarts, and deployments. Not a task queue, not a state machine — it's a new programming paradigm where you write business logic as straightforward sequential code and the platform handles reliability.

**History**: Maxim Fateev & Samar Abbas built Amazon SWF (2009) → Uber's Cadence (2015) → Temporal (2019). Three decades of workflow orchestration expertise. Series D at $5B valuation.

**Business model**: Open-source server (MIT) + Temporal Cloud (consumption-based: ~$25/million actions). Revenue comes from Cloud hosting, support, enterprise features. Available on AWS, GCP, Azure Marketplaces.

**vs competitors**: AWS Step Functions (JSON-based, limited duration, vendor lock-in), Azure Durable Functions (C#/.NET focused), Apache Airflow (DAG-based, not general-purpose), Inngest/Hatchet (newer, less battle-tested). Temporal's key differentiators: code-first, unlimited duration, replay-based debugging, multi-language SDKs, worker-based versioning.

### Server Architecture

```
Clients → Frontend Service (stateless API gateway, auth, rate limiting, routing)
              ↓
         ┌────┴────┐
         ↓         ↓
    History      Matching
    Service      Service
    (workflow    (task queue
    state,       dispatch,
    sharding     sync matching)
    by WF ID)
         ↓         ↓
         └────┬────┘
              ↓
       Persistence Layer
       (Cassandra/MySQL/PostgreSQL + Elasticsearch for visibility)
```

- **Frontend**: Stateless gRPC gateway. Rate limiting, auth, request routing.
- **History**: Owns workflow state. Shards by workflow ID (512 default). Serializes operations per shard = strong consistency. This is the brain.
- **Matching**: Dispatches tasks to workers via long-polling. Sync matching optimization: if a worker is already waiting, task delivered immediately without DB round-trip.
- **Worker**: Customer-hosted processes that execute workflow/activity code. Poll task queues. Stateless — all state is in the server's event history.

### How Durable Execution Works

1. Workflow code executes on a worker. Each completed step (activity, timer, child workflow) generates an **event** appended to the workflow's **event history**.
2. If the worker crashes, a new worker picks up the workflow. The SDK **replays** the event history through the workflow function, returning previously recorded results for completed steps.
3. Replay reconstructs in-memory state without re-executing side effects. The workflow resumes exactly where it left off.
4. **Determinism requirement**: Replay must produce the same command sequence. No `time.Now()`, `rand`, environment reads, or direct I/O in workflow code. Use SDK-provided alternatives.

### Values (Know These for Behavioral Questions)

**Curious** — Dig into how things work, not just how to use them. Read source code, investigate root causes.
**Driven** — Own outcomes end-to-end. Ship with clear milestones.
**Collaborative** — Align teams via design docs. Incorporate cross-functional expertise.
**Genuine** — Admit unknowns. Own mistakes in postmortems. Give direct feedback.
**Humble** — Credit others. Change your mind on evidence. Listen before assuming.

---

## 2. Temporal SDK Patterns

### Core Primitives

| Concept | What It Is | Interview Tip |
|---------|-----------|---------------|
| **Workflow** | Deterministic orchestrator function. Survives any failure via replay. | Emphasize: "reliable function execution" not "state machine" |
| **Activity** | Side-effectful operation (API calls, DB writes). Retried independently per policy. | Know retry policy fields: initial interval, backoff coefficient, max interval, max attempts, non-retryable errors |
| **Worker** | Process that polls a task queue and executes workflows/activities. Stateless. | Workers are your code running on your infra. Temporal server never executes your business logic. |
| **Task Queue** | Named queue connecting tasks to workers. Enables routing and versioning. | "GPU tasks to GPU workers" is the canonical example |
| **Namespace** | Isolation unit. Separate event histories, task queues, visibility. | Temporal Cloud bills per namespace |

### Signals, Queries, Updates

- **Signal**: Async durable message to a running workflow. Fire-and-forget. The workflow receives it on next replay. Use for: "customer upgraded plan", "payment received".
- **Query**: Sync read-only inspection of workflow state. No side effects, no history events. Use for: "what's the current invoice status?".
- **Update**: Sync validated mutation with a response. Validates input, mutates state, returns result. Use for: "apply coupon code, return new price".

### Continue-as-New

Completes current execution and immediately starts a new one with fresh event history, same workflow ID. Prevents unbounded history growth. Essential for long-running workflows (subscriptions, monitors, billing cycles). Pass accumulated state as input to the new execution.

### Saga Pattern with Temporal

```
// Compensating activities in reverse order on failure
err := workflow.ExecuteActivity(ctx, CreateInvoice, ...).Get(ctx, &invoice)
if err != nil { return err }

err = workflow.ExecuteActivity(ctx, ChargePayment, ...).Get(ctx, &payment)
if err != nil {
    // Compensation: void the invoice
    _ = workflow.ExecuteActivity(ctx, VoidInvoice, invoice.ID).Get(ctx, nil)
    return err
}
```

### Workflow Versioning

```go
v := workflow.GetVersion(ctx, "billing-v2", workflow.DefaultVersion, 1)
if v == workflow.DefaultVersion {
    // Old code path for in-flight workflows
} else {
    // New code path for new executions
}
```

Records a version marker in history. Old executions take old path; new executions take new path. Safe code changes for in-flight workflows. Phase out old code once all existing workflows complete.

### Billing-Specific Temporal Patterns

- **Subscription lifecycle**: Long-running workflow per customer. Signals for plan changes, payment events. Continue-as-new each billing period.
- **Invoice generation**: Child workflow per invoice. Activities: calculate usage → apply pricing → generate line items → finalize → send to payment processor.
- **Metering aggregation**: Workflow per namespace per hour. Durable execution guarantees no missed aggregation windows.
- **Dunning**: Workflow started on payment failure. Timer-based retry schedule (day 0, 3, 7, 10). Signal to resume on payment method update.

---

## 3. Billing & Metering Systems

### Architecture

```
Usage Events → Metering Pipeline → Aggregation → Rating → Invoicing → Payment
```

**Metering**: Collect raw usage events (workflow starts, activity executions, signals, storage). Dedup with idempotency keys. Archive to S3 (Parquet). Stream through Kinesis.

**Aggregation**: Hourly rollups per namespace per action type. Temporal workflows ensure no missed windows. Idempotent: recompute from source, never increment.

**Rating**: Apply pricing tiers to aggregated usage. Per-unit pricing with volume discounts. Handle plan changes mid-cycle (proration).

**Invoicing**: Generate invoice at period end. Draft → finalize (48hr grace for late events) → send to payment processor → paid/failed.

**Payment**: Stripe for direct. AWS/GCP/Azure Marketplace for marketplace customers. Adapter pattern abstracts the payment channel.

### Key Principles

- **Integer money**: Store as BIGINT cents. $10.50 = 1050. Never float/decimal for money in transit.
- **Idempotency everywhere**: Unique keys on every mutation. Stripe supports idempotency_key natively (24hr TTL).
- **Double-entry bookkeeping**: Every transaction = debit + credit. Total debits must equal total credits. Self-balancing, auditable, reversals are explicit new entries.
- **Immutable events**: Raw metering events are append-only. Never update. Source of truth for reconciliation.
- **Reconciliation**: Hourly lightweight counts. Daily per-tenant detail. Monthly full reconciliation before billing close.

### Stripe Integration

- We own billing logic; Stripe is the payment processor. We create invoices and charge; we don't derive billing state from Stripe.
- Webhooks: verify signature → return 200 immediately → process async via Temporal workflow (durable, retryable).
- Key events: `invoice.paid`, `invoice.payment_failed`, `customer.subscription.updated`, `charge.refunded`.
- Proration: calculate ourselves (days remaining × price difference / total days). Log calculation for audit.

### Marketplace Integration

**AWS Marketplace**: Customer subscribes → fulfillment URL → `ResolveCustomer` → `GetEntitlements` → onboard. Report usage via `BatchMeterUsage` hourly. **Critical**: must report within 6 hours or lose billing for that hour.

**GCP Marketplace**: Procurement API with approval workflow. Must actively approve/reject subscriptions. Entitlement states: PENDING, ACTIVE, SUSPENDED, CANCELLED.

**Adapter pattern**: `BillingChannel` interface with `OnboardCustomer`, `ReportUsage`, `ProcessPlanChange`, `GetExternalState`. Implementations: `StripeChannel`, `AWSMarketplaceChannel`, `GCPMarketplaceChannel`. All business logic in the billing service, not adapters.

### Dunning Flow

Day 0: Payment fails → PAST_DUE → email. Day 3: retry. Day 7: retry + final warning. Day 10: retry → SUSPENDED if still failing. Day 30: CANCELLED. At any point: signal on payment method update → immediate retry.

---

## 4. Go for Systems Engineering

> See also: `courses/golang/` for language fundamentals.

### Concurrency

- **Goroutines**: User-space, ~2KB initial stack, M:N scheduled. Run millions. Not OS threads.
- **Channels**: Typed, synchronized. Unbuffered blocks sender until receiver ready. Buffered blocks when full.
- **`select`**: Multiplexes across channels. Like `Promise.race`.
- **`context.Context`**: Carries deadlines, cancellation, request-scoped values. First parameter of every blocking function. Always respect cancellation.
- **`sync.WaitGroup`**: Wait for N goroutines to complete. `Add(n)` before launching, `Done()` in each goroutine, `Wait()` to block.

### Error Handling

- Errors are values, returned explicitly. `if err != nil` at every call site.
- Wrap: `fmt.Errorf("creating invoice: %w", err)`. Check: `errors.Is(err, ErrNotFound)`, `errors.As(err, &myErr)`.
- Reserve `panic` for programmer bugs only. Never for expected failures.

### Interfaces

- Implicitly satisfied — no `implements` keyword. Define at the consumer, not the producer.
- Keep small: 1-3 methods. `io.Reader`, `io.Writer` are the gold standard.
- Use for testing: define interface, mock implementation, inject dependency.

### Database Patterns

- `database/sql` with connection pooling. Set `MaxOpenConns`, `MaxIdleConns`, `ConnMaxLifetime`.
- Always `defer tx.Rollback()` after `BeginTx()` (no-op if committed).
- Use `SERIALIZABLE` isolation for billing transactions.

### Testing

- Table-driven tests with subtests (`t.Run`).
- `-race` flag catches data races at runtime. Use in CI.
- Mock via interfaces — no frameworks needed.

---

## 5. Data Systems

> See also: `courses/infrastructure/02-databases-at-scale/` and `courses/infrastructure/04-message-queues/`.

### Storage System Roles

| System | Role in Billing | Best For |
|--------|----------------|----------|
| **PostgreSQL** | Events, invoices, ledger (OLTP) | Transactions, single-row ops, < 10ms |
| **Redshift** | Usage reports, dashboards (OLAP) | Aggregation over billions of rows |
| **Kinesis** | Event ingestion stream | Real-time ingestion, milliseconds |
| **S3** | Raw event archive, data lake | Durable archival, reprocessing |

### PostgreSQL at Scale

- **Indexing**: B-tree for equality/range. GIN for JSONB/arrays. Partial indexes for subsets. Covering indexes (INCLUDE) for index-only scans.
- **Partitioning**: Table > 100M rows → PARTITION BY RANGE (created_at) monthly. Enables partition pruning and clean archival (detach + drop).
- **Key thresholds**: max_connections 100-500, shared_buffers 25% RAM, target cache hit ratio > 99%.

### Kinesis

- 1 MB/s write per shard, 2 MB/s read, 1000 records/s, max 1 MB record.
- Consumer checkpointing: process batch → write to DB with idempotency keys → checkpoint. Never checkpoint before processing completes.
- Hot shard: split shard, or add random suffix to partition key (sacrifices ordering).

### Pipeline Patterns

**Idempotent aggregation** (the critical pattern):

```sql
-- WRONG: increment is NOT idempotent (replay doubles the count)
UPDATE usage_summaries SET total = total + $1 WHERE ...;

-- RIGHT: recompute from source (replay produces same result)
INSERT INTO usage_hourly (tenant_id, event_type, hour, total_quantity)
SELECT tenant_id, event_type, date_trunc('hour', event_timestamp), SUM(quantity)
FROM metering_events WHERE ...
GROUP BY tenant_id, event_type, date_trunc('hour', event_timestamp)
ON CONFLICT (tenant_id, event_type, hour)
DO UPDATE SET total_quantity = EXCLUDED.total_quantity;
```

**Outbox pattern**: Write business data + event to outbox table in one transaction. Separate process publishes from outbox to Kinesis. Avoids dual-write inconsistency.

**CDC**: PostgreSQL WAL → logical replication slot → Debezium → Kafka/Kinesis → S3/Redshift. Monitor `pg_replication_slots` for WAL lag.

### Reconciliation

- **Hourly**: lightweight count comparison (PostgreSQL vs Redshift).
- **Daily**: per-tenant, per-event-type detail.
- **Monthly**: full reconciliation before billing close.
- PostgreSQL is always authoritative. Redshift/S3 lag 5-15 minutes.

---

## 6. Platform Architecture (Cloud Nexus)

**Cloud Nexus**: Self-service API for connecting Temporal namespaces across clouds, regions, and teams.

### Key Concepts

**Control Plane** (global): Registry, policies, routing rules, provisioning. Consistency > latency. Backed by PostgreSQL.

**Data Plane** (per-region): Stateless Nexus proxies that route requests, enforce policies, and load balance. Latency-critical. Caches control plane config locally. **Must function even if control plane is down** (cached policies/routes).

**Access Control** (3 layers):
1. Connection approval — human-in-the-loop for cross-org, auto-approve intra-org.
2. Policy evaluation — cached at data plane (30s TTL), < 1ms evaluation.
3. mTLS enforcement — unique cert per namespace, prevents impersonation.

**Tenant isolation**: Per-namespace rate limits, per-org quotas, org-scoped metrics/logs. No cross-tenant data leakage.

**Multi-region routing**: Route tables computed by control plane, pushed to proxies. Active + passive health checks. Failover to secondary routes. Latency-based routing for multi-region namespaces.

**Onboarding**: Temporal workflow orchestrates provisioning (validate → allocate infra → register namespaces → enable billing → welcome notification). Saga-pattern compensation on failure.

---

## 7. Interview Prep

### Quick-Fire Technical Answers

**Temporal**
1. **What is Temporal?** Durable execution platform — write workflows as code, guaranteed to complete through failures.
2. **How does replay work?** SDK replays event history through the function, returning cached results for completed steps, reconstructing state without re-executing side effects.
3. **Why deterministic?** Replay must produce the same command sequence. Non-deterministic code diverges from event history.
4. **Signals vs queries vs updates?** Signals: async durable messages. Queries: sync read-only. Updates: sync validated mutations with response.
5. **Continue-as-new?** Complete and restart with fresh history, same workflow ID. Prevents unbounded history.
6. **What is a Nexus endpoint?** Cross-namespace/cross-cluster communication primitive. Typed, discoverable API boundary between namespaces.

**Distributed Systems**
7. **Exactly-once delivery?** Impossible. Achieve effectively-exactly-once with at-least-once delivery + idempotent processing.
8. **CAP theorem for billing?** Choose consistency — wrong charges are worse than downtime.
9. **Saga pattern?** Local transactions with compensating actions. Temporal workflows are a natural fit.
10. **Event sourcing?** Append-only events as source of truth. Rebuild state by replaying. Temporal's event history is exactly this.

**Billing**
11. **Double-entry bookkeeping?** Every transaction = debit + credit. Debits must equal credits. Catches errors, prevents silent corruption.
12. **Why integer money?** Floats can't represent 0.1 exactly. Use BIGINT cents.
13. **Idempotency keys?** Unique ID per mutation. Same key = return original result without re-executing.
14. **Webhook reliability?** At-least-once. Verify signature, return 200 fast, process async, deduplicate by event ID, reconciliation as safety net.

### System Design Exercises

**Exercise 1 — Metering Pipeline**: Design for 50M events/day. Kinesis → S3 archive (Firehose) + aggregation workers (Temporal workflows, hourly per namespace) → PostgreSQL → billing service. Dedup: bloom filter + DB UNIQUE. Late events: 5min grace period, daily correction workflow, post-finalization adjustments. Dashboard: separate Redis consumer for real-time.

**Exercise 2 — Billing with Marketplaces**: Billing service owns canonical state; Stripe/AWS/GCP are payment channels via adapter pattern. Stripe: we create invoices, charge via API. AWS: `BatchMeterUsage` hourly (6hr deadline). Reconciliation: daily per-channel state comparison. Customer lifecycle state machine: TRIAL → ACTIVE → PAST_DUE/UPGRADING/CANCELLING → CANCELLED.

**Exercise 3 — Cloud Nexus**: Control plane (registry, policies, routing) + data plane (stateless proxies per region). Connection approval + policy evaluation + mTLS enforcement. Data plane caches config, survives CP outage. Onboarding as Temporal workflow with saga compensation.

### System Design Template (45 min)

1. **Clarify requirements** (3 min) — Ask about scope, scale, latency, consistency. State assumptions.
2. **High-level architecture** (5 min) — Boxes and arrows. Name components. Explain decisions as you draw.
3. **Deep dive** (15-18 min) — Pick 1-2 components. Data model, API, algorithms, error handling.
4. **Tradeoffs** (3-5 min) — "I chose X over Y because Z. If we needed A, we'd switch to Y."
5. **Operational concerns** (3-5 min) — Monitoring, deployment, failure modes, 10x scaling, on-call experience.

### Behavioral Story Bank

Prepare 1-2 STAR stories per value. Practice each in under 2 minutes.

| Value | Story Theme |
|-------|-------------|
| **Curious** | Learned a system by reading source code / investigated a performance regression by profiling |
| **Driven** | Shipped a migration under tight deadline / took ownership of a system no one wanted |
| **Collaborative** | Worked with non-eng team on requirements / aligned three teams on shared API via RFC |
| **Genuine** | Owned a production incident in postmortem / gave direct feedback to a peer |
| **Humble** | Changed architecture based on junior's insight / spent first month on new team listening |

### Questions to Ask

**Team**: How is the billing team structured? What does on-call look like? How does the team interact with the core server team?

**Cloud Nexus**: Current state (preview/GA)? Biggest technical challenges? Customer adoption patterns?

**Billing Platform**: Current stack (Stripe, custom, mix)? Biggest source of billing incidents? Planned migrations?

**Culture**: Design review process (RFCs, ADRs)? Deployment frequency and pipeline? Incident response process?

**Growth**: Where is the billing platform in 12 months? What does 6-month success look like? How do you prioritize features vs reliability?

### Red Flags to Avoid

- Not knowing what Temporal does or why determinism matters
- Jumping into solution mode without clarifying requirements
- Using floating-point for money, ignoring deduplication or reconciliation
- Designing without discussing failure modes
- Claiming Go experience without knowing error wrapping, interfaces, or goroutines
- Saying "exactly-once delivery" as if it exists
- Blaming others, no specific examples, not asking questions at the end

### Key Numbers

| System | Metric | Value |
|--------|--------|-------|
| Temporal | Event history max | 50K events or 50MB per execution |
| Temporal | Signal delivery | < 100ms p99 |
| Stripe | API rate limit | 100 read + 100 write req/s per key |
| Stripe | Idempotency key TTL | 24 hours |
| Stripe | Webhook retries | 16 retries over ~3 days |
| Kinesis | Write per shard | 1000 records/s or 1 MB/s |
| Kinesis | Read per shard | 5 txn/s, 2 MB/s |
| PostgreSQL | Transaction throughput | 10K-50K simple TPS |
| PostgreSQL | Connection overhead | ~10MB RAM per connection |

---

## Day-Before Checklist

- [ ] Review Temporal docs: workflows, activities, workers, signals, queries, continue-as-new
- [ ] Re-read the job description. Map keywords to your experience.
- [ ] Prepare 5 behavioral stories in STAR format (< 2 min each)
- [ ] Walk through 1 system design out loud (metering pipeline is most likely). Time yourself.
- [ ] Prepare 5 questions to ask the interviewer
- [ ] Review Go concurrency: goroutines, channels, select, sync.WaitGroup, context
- [ ] Review Stripe basics: subscriptions, invoices, webhooks, idempotency keys
- [ ] Test setup: camera, mic, screen sharing, IDE
- [ ] Sleep. A well-rested brain outperforms a cramming brain every time.
