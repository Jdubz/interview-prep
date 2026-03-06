# Module 01: High-Volume Scaling Experience

## Overview

Theoretical scaling knowledge and production scaling experience feel identical on paper — and very different in an interview room. Engineers who have operated high-volume systems talk about failure first, think in percentiles instead of averages, and describe decisions with before/after context rather than abstract trade-offs. This module builds that experiential layer: the real failure patterns that break systems, how to frame decisions the way practitioners do, and the vocabulary that signals genuine depth.

The gap this module closes: you know *what* consistent hashing is. This teaches you to sound like someone who has actually dealt with a hot partition at 3am.

---

## Part 1: The Mental Model of Scale

### Thinking From Failure, Not Success

Engineers without scaling experience think about systems **when they work**. Engineers who have operated high-volume systems think about systems **when they fail at 100x load**. That mental inversion is the single clearest signal interviewers look for.

**What this sounds like in practice:**

Without experience: "We'll put a cache in front of the database to reduce read latency."

With experience: "We'll add Redis with cache-aside. The tricky failure mode is cache stampede -- if the cache cold-starts or all TTLs expire simultaneously, 50K concurrent requests hit the database at once. We handle this with jittered TTLs: instead of a fixed 10-minute expiry, we randomize to 8-12 minutes so expirations stagger naturally."

The content is nearly identical. The difference is that experienced engineers *lead with the failure mode*.

### Thinking in Percentiles, Not Averages

Average latency is almost useless for understanding user experience. A system averaging 50ms with a p99 of 5 seconds is a bad system that looks fine in dashboards.

```
p50  = the median. 50% of requests are faster than this.
p95  = 95% of requests are faster than this.
p99  = 99% of requests are faster than this.
p999 = 99.9% of requests are faster than this.
```

In a system handling 10,000 RPS:
- p99 = 1% of requests = 100 requests/second are at or above this latency
- p999 = 0.1% of requests = 10 requests/second

At scale, even p999 latencies are experienced by real users constantly. Design accordingly.

**In interviews**: State your non-functional requirements in percentiles, not averages. "I want p99 under 200ms" is specific and measurable. "I want low latency" is not.

### Orders of Magnitude Thinking

Scale does not change systems linearly. Going from 1K QPS to 10K QPS is a different problem than going from 10K to 100K. Know which multiplier you are in:

| Scale | Typical Solution |
|---|---|
| 1 - 1K QPS | Single server, single database |
| 1K - 10K QPS | Read replicas, application caching |
| 10K - 100K QPS | Dedicated cache tier, connection pooling, CDN |
| 100K - 1M QPS | Database sharding, microservices, regional distribution |
| 1M+ QPS | Custom infrastructure, edge computing, protocol-level optimization |

The interview anti-pattern is jumping to the 1M+ solution for a system that needs the 1K one. Let estimation drive complexity.

---

## Part 2: Real Production Failure Patterns

These are the failure modes that actually bring down production systems at scale. Knowing them signals experience. Being able to describe mitigation signals seniority.

### Hot Partition / Hot Key

**What it is**: A sharding or partitioning strategy that distributes load evenly on average but concentrates traffic on a single shard under real access patterns.

**Classic scenario**: You shard a social platform by `user_id`. A celebrity with 50M followers joins. Every follower's feed read, every notification, and every analytics event touches that user's data. That shard now handles 20% of cluster traffic.

**Detection**: Uneven CPU or I/O utilization across shards. Latency spikes from one partition while others are idle.

**Mitigations**:
- **Write sharding (key suffixing)**: Append a random suffix to hot keys, e.g., `user:12345:shard:3`. Distribute across N virtual partitions. Reads must query all N and merge.
- **Composite shard keys**: Include a secondary dimension (e.g., `user_id + date`) to spread temporally.
- **Read-through cache with coalescing**: Cache the entity at the application layer so multiple in-flight requests for the same key are collapsed into one backend call.
- **Celebrity tier**: Identify entities above a threshold (follower count, request rate) and route them to a special handling path with different infrastructure.

### Thundering Herd / Cache Stampede

**What it is**: Multiple cache entries expire simultaneously, causing a burst of backend requests before the cache is repopulated.

**Classic scenario**: Your product catalog is cached with a fixed 10-minute TTL. You deploy a new version at 2:00pm. The cache warms over the next 10 minutes. At 2:10pm, every cache entry expires simultaneously. 40,000 concurrent requests hit the database at once. The database falls over.

**Detection**: Periodic database CPU spikes at predictable intervals, correlated with cache TTLs. Often a deployment or cache restart is the trigger.

**Mitigations**:
- **Jittered TTL**: `TTL = base_ttl + random(0, base_ttl * 0.2)`. Stagger expirations across a window.
- **Mutex/lock on cache miss**: The first request to miss grabs a distributed lock and fetches from the database. All other concurrent misses wait for the lock holder to repopulate the cache.
- **Probabilistic early rehydration (PER)**: Before a cache entry expires, a fraction of requests trigger background refresh with probability inversely proportional to remaining TTL. Entries are refreshed before they expire.
- **Cache-warming on deploy**: Proactively populate the cache during deployment before routing live traffic.

### The N+1 Query Problem at Scale

**What it is**: Loading N entities then issuing a separate query for each one, resulting in N+1 total queries instead of 1.

**Classic scenario**:
```
// Load 100 users
users = db.query("SELECT * FROM users LIMIT 100")

// For each user, load their latest post -- 100 additional queries
for user in users:
    user.latest_post = db.query("SELECT * FROM posts WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", user.id)
```

At 100 users this adds 100ms overhead. At 10K concurrent requests each fetching 100 users, this is 1 million database queries per second from a single endpoint.

**Detection**: Database query count spikes proportionally with entity count, not request count. `EXPLAIN ANALYZE` shows sequential single-row lookups.

**Mitigations**:
- **Batch queries**: Replace the loop with `WHERE id IN (id1, id2, ..., idN)`.
- **JOIN-based eager loading**: Fetch related data in the original query with a LEFT JOIN.
- **DataLoader pattern**: Batch and deduplicate all database requests within a single request cycle. Originally from GraphQL but applicable anywhere.
- **Denormalization**: Store `latest_post_id` on the user row to eliminate the secondary lookup entirely.

### Connection Pool Exhaustion

**What it is**: The application opens more database connections than the pool allows (or the database can handle), causing requests to queue waiting for a connection.

**The numbers**: Each PostgreSQL connection consumes ~10MB of memory on the database server. A database with 8GB RAM and typical settings handles ~800-1000 connections. An application with 50 instances each configured for a pool of 25 connections makes 1,250 potential connections -- already at the limit.

**Classic scenario**: A traffic spike doubles concurrent requests. Connection checkout time goes from <1ms to 2 seconds. All requests time out waiting for a connection. The incident looks like the database is slow, but the database itself is healthy.

**Detection**: `pg_stat_activity` shows max connections used. Application metrics show elevated `connection_wait_time`. Database CPU is low while requests queue.

**Mitigations**:
- **Connection pooler (PgBouncer)**: Sits between app and database, multiplexes many app connections to fewer database connections. Transaction-mode pooling allows 1000 app connections to share 20 database connections.
- **Limit pool size per instance**: `pool_size = (total_db_connections - maintenance_connections) / num_app_instances`. Most apps need far fewer connections per instance than developers assume.
- **Async I/O**: Async runtimes (Go, Node.js, async Python) can handle many concurrent requests per connection. Synchronous thread-per-request models (traditional Java, Rails) require one connection per concurrent request.
- **Circuit breaker on pool exhaustion**: Fail fast with a 503 when the pool is exhausted rather than queuing requests that will all timeout.

### Write Amplification

**What it is**: A single logical write operation triggers multiple physical write operations internally, consuming more I/O than expected.

**Where it appears**:
- **LSM-tree databases (Cassandra, RocksDB, DynamoDB)**: Data is first written to a memtable (in memory) and an append-only commit log, then flushed to SSTables on disk. Background compaction reads and rewrites SSTables to merge data and reclaim space. A single write can be physically written 5-30x before it settles.
- **Databases with many indexes**: A table with 8 indexes means each row insert or update writes to 9 structures (the table heap + 8 index trees).
- **Replication**: Each write to the primary must be replicated to N followers.

**Detection**: Write I/O throughput is much higher than the logical write rate would suggest. Database write latency increases under sustained write load. Compaction storms cause unpredictable latency spikes.

**Mitigations**:
- **Tune compaction aggressively**: More frequent compaction reduces write amplification at the cost of I/O during quiet periods (STCS vs. LCS in Cassandra).
- **Drop unused indexes**: Every index is a write amplifier. Audit and remove indexes that are not serving real queries.
- **Batch writes**: Amortize the overhead across many logical writes. Writing 1000 records as a batch is often faster than 1000 individual writes.
- **SSDs**: Write amplification is much less damaging on SSDs than HDDs because SSDs handle random writes well and have their own internal write buffering.

### The Fan-Out Problem

**What it is**: A single event (a post, a message, an action) must be delivered to many consumers, and the delivery mechanism does not scale with the consumer count.

**Classic scenario**: Twitter's "celebrity tweet" problem. A user with 50 million followers posts a tweet. If you fan-out to all follower timelines synchronously, you make 50 million database writes in a single request. If you compute feeds on read, each timeline read requires aggregating posts from everyone a user follows.

**The fundamental trade-off**: Fan-out on write (push) vs. fan-out on read (pull).

```
Fan-out on write (push):
  Write:  O(followers) -- expensive at write time
  Read:   O(1) -- just read pre-assembled timeline
  Best for: users with few followers, read-heavy workloads

Fan-out on read (pull):
  Write:  O(1) -- just write the post
  Read:   O(following_count) -- assemble at read time
  Best for: users with many followers, write-heavy workloads
```

**Real-world solution (Twitter/X hybrid)**:
- Users below ~10,000 followers: fan-out on write. Their tweets are pushed to follower timelines at write time.
- "Celebrities" above the threshold: fan-out on read. Their posts are not pushed; each follower's timeline assembly fetches their latest posts at read time.
- Timeline assembly merges the pre-assembled timeline (from write fan-out) with a real-time fetch of celebrities the user follows.

**The interview insight**: Most real systems use a hybrid. The threshold between write fan-out and read fan-out is a tunable parameter, not a binary architectural choice.

---

## Part 3: Speaking From Experience

### The Narrative Pattern

Production experience has a structure that is recognizable and credible: **before state → inflection point → decision with trade-offs → after state**.

Construct answers in this shape even when describing hypothetical systems:

**Without experience**: "Caching can help with read performance. You can use Redis with cache-aside to reduce database load."

**With experience**: "The product catalog endpoint was at ~15K QPS, all hitting Postgres. At that load, p99 started creeping past 500ms, which violated our SLO. We introduced Redis with cache-aside and a 5-minute TTL. Database load dropped 80%. The main complexity was invalidation -- we moved from TTL-only to event-driven invalidation from the write path, so product updates reflected in under a second instead of up to 5 minutes. The risk we accepted: the write path now has a Redis dependency, so we designed the write to succeed even if the cache invalidation fails -- just accept up to 5 minutes of staleness."

Same information. The second version sounds like it happened. Use this structure.

### Vocabulary Signals

Words and phrases that signal operational experience:

| Phrase | What It Signals |
|---|---|
| "p99 latency" | Thinks in percentiles, not averages |
| "our SLO at the time was..." | Operational awareness, clarity on what matters |
| "the blast radius" | Thinks about failure scope |
| "we accepted the risk of..." | Acknowledges trade-offs explicitly |
| "the first thing that broke was..." | Has seen systems fail sequentially |
| "we observed in production..." | Experience-grounded, not theoretical |
| "we set an error budget" | Understands reliability engineering |
| "the cardinality of..." | Precise thinking about data dimensions |
| "at that write amplification factor" | Understands physical vs. logical I/O |
| "we colocate these writes" | Thinks about data locality |

### Recognizing Scale Inflection Points

Experienced engineers know which metric signals each problem is coming before it arrives. Name the inflection point and the signal:

| Problem | Early Warning Signal |
|---|---|
| Database becoming bottleneck | p99 read latency increasing, CPU > 60% on DB |
| Cache stampede risk | Fixed TTLs on high-traffic keys after a deploy |
| Hot partition emerging | Uneven I/O distribution across shards in monitoring |
| N+1 regression | DB query count growing superlinearly with request count |
| Connection pool saturation | `connection_wait_time` > 0 in app metrics |
| Fan-out too expensive | Write latency correlated with author follower count |
| Write amplification | DB write I/O >> logical write rate |

---

## Part 4: The Scaling Progression

Real systems do not start at Netflix scale. They evolve through identifiable stages. Knowing this progression lets you anchor interview answers to the right complexity level and demonstrate that you do not over-engineer.

```
Stage 1: Single Server
  Everything on one machine: app, database, file storage.
  Limit: ~1K concurrent users, ~500 QPS.
  First thing to break: database becomes bottleneck under read load.
  Next step: add a read replica.

Stage 2: Read Replicas
  Primary handles writes. One or more replicas handle reads.
  Application routes reads to replicas, writes to primary.
  Limit: ~5K-10K QPS reads (add replicas), write throughput capped by primary.
  Problem: replication lag causes stale reads. Handle with read-your-own-writes.
  First thing to break: write throughput on primary, or replica count becomes unwieldy.
  Next step: add a caching layer.

Stage 3: Caching Layer
  Redis or Memcached in front of the database.
  Cache-aside for most use cases, write-through for strong consistency.
  Limit: ~50K-100K QPS (cache hit rate dependent).
  Problem: cache invalidation, thundering herd, hot keys.
  First thing to break: memory limits, hot key skew.
  Next step: horizontal app scaling.

Stage 4: Horizontal App Scaling + Load Balancer
  Multiple stateless app servers behind a load balancer.
  Requires externalizing all state: sessions to Redis, uploads to S3.
  Limit: scales horizontally until database is the bottleneck again.
  Problem: stateful operations (WebSockets, file uploads) need special handling.
  First thing to break: database write throughput or storage limits.
  Next step: database sharding or CQRS.

Stage 5: Database Sharding
  Data partitioned across multiple database nodes by shard key.
  Each shard handles a subset of writes and reads.
  Limit: near-infinite horizontal scale with proper shard key selection.
  Problem: cross-shard queries, distributed transactions, operational complexity.
  First thing to break: cross-shard query patterns you did not anticipate.
  Next step: domain-driven service boundaries.

Stage 6: Microservices + Event-Driven Architecture
  Services split by domain, each with its own database.
  Communication via events (Kafka) or APIs (gRPC/REST).
  Limit: organizational and operational complexity, not technical.
  Problem: distributed tracing, saga coordination, eventual consistency.
  First thing to break: your ability to debug and reason about the system.
  Next step: platform engineering, service mesh, observability.

Stage 7: Global Distribution
  Multi-region active-active or active-passive.
  Traffic routed by geography, data replicated across regions.
  Limit: CAP theorem constraints, data sovereignty, conflict resolution complexity.
  Problem: latency vs. consistency, GDPR data residency requirements.
```

**Interview application**: When asked to design something, start at the appropriate stage based on the stated scale, then explicitly describe how you would evolve the design if traffic grew 10x.

---

## Part 5: Interview Evaluation Rubric

What interviewers score and what differentiates levels. Use this to self-assess your answers.

| Signal | Junior | Senior | Staff |
|---|---|---|---|
| **Estimation** | Can do it when prompted | Does it unprompted, accurately | Uses it to challenge requirements |
| **Failure modes** | Rarely mentioned | Identifies 2-3 key failures | Designs for failure from the start |
| **Trade-offs** | Knows they exist | States them before committing | Weights them against business constraints |
| **Alternatives** | Picks one approach | Compares 2-3 and commits | Knows when NOT to use standard patterns |
| **Operational concerns** | Not mentioned | Brief mention | SLOs, monitoring, and alerting as first-class |
| **Scaling path** | Single design only | Notes 10x considerations | Builds evolution path into the design |
| **Vocabulary** | Conceptual terms | Precise operational terms | Uses terms with quantified context |

**The clearest senior signal**: Proactively identifying the hardest part of the system before the interviewer asks, then proposing a deep dive on it. "The challenging piece here is fan-out at scale -- let me walk through the trade-offs between write fan-out and read fan-out and where the threshold should be."

**The clearest staff signal**: Reframing the problem. "The way this is stated, we're building for 10M users. But if this is a social platform, the real constraint is user distribution -- 10M evenly distributed users is a very different problem than 10M where 100 accounts generate 40% of content. Let me clarify the access pattern before I commit to a sharding strategy."

---

## References

These resources are the standard references for experienced engineers. Knowing what they contain -- even without reading them cover to cover -- is valuable.

### Books

**Designing Data-Intensive Applications** by Martin Kleppmann
The definitive reference for everything in this module. Covers storage engines, replication, partitioning, distributed transactions, batch and stream processing. Read Chapters 5-9 for the core distributed systems material. This is the book interviewers reference when they ask about replication lag, CAP theorem, or consistency models.

**System Design Interview volumes 1 and 2** by Alex Xu
The canonical interview preparation resource. Less rigorous than DDIA but extremely focused on interview context. Volume 1 covers the foundational 12 design problems. Volume 2 adds payment systems, metrics collection, and distributed email.

**The Art of Scalability** by Abbott and Fisher
Introduces the AKF Scale Cube: scale on X axis (horizontal duplication), Y axis (functional decomposition), Z axis (data partitioning). Useful for articulating *which* kind of scaling you are applying and why.

### Online Resources

**Hello Interview — System Design Patterns**
`hellointerview.com/learn/system-design/patterns`
Focused articles on scaling reads, scaling writes, and consistent hashing. Written specifically for interview preparation with rubric awareness. The "Scaling Reads" and "Scaling Writes" guides are directly interview-applicable.

**GitHub: System Design Primer** by Donne Martin
`github.com/donnemartin/system-design-primer`
The most-referenced open-source system design resource. Covers every component type with diagrams, trade-offs, and links to primary sources. Use as a reference when you want to verify your understanding of a specific component.

**High Scalability Blog**
`highscalability.com`
Case studies of real production architectures: how Twitter, Uber, Discord, and others evolved their systems. Reading 5-10 of these gives you concrete examples to reference: "Discord moved from MongoDB to Cassandra for their message store when they hit 100M messages per day, primarily because of..."

**Netflix Tech Blog**
`netflixtechblog.com`
Primary source for multi-region active-active architecture, chaos engineering (Chaos Monkey), and stream processing at scale. Reference when discussing availability patterns and failure injection.

**AWS Architecture Blog**
`aws.amazon.com/blogs/architecture`
Practical guides for patterns in AWS: how DynamoDB handles partitioning, how SQS handles at-least-once delivery, how ElastiCache fits into scaling strategies.

### Specific Articles Worth Reading

- "Dynamo: Amazon's Highly Available Key-Value Store" (2007 paper) — the origin of the AP distributed database model and consistent hashing at production scale.
- "The Log: What every software engineer should know about real-time data's unifying abstraction" by Jay Kreps (LinkedIn, 2013) — the conceptual foundation for Kafka, event sourcing, and the log as a first-class data structure.
- "CAP Twelve Years Later: How the Rules Have Changed" by Eric Brewer (2012) — Brewer's own clarification of CAP theorem, which is more nuanced than the simple triangle suggests.

---

## Interview Questions

Questions are organized by type and difficulty. Each mirrors what interviewers actually ask. Answer by speaking through your reasoning, not by reciting memorized answers.

### Estimation Questions

**Q1. A social media platform has 50M DAU. Each user views their feed 5 times per day. Each feed shows 20 posts. Estimate the read QPS for the feed service.**

Expected approach and answer:
```
Feed views/day = 50M users * 5 views/user = 250M views/day
Posts fetched/day = 250M views * 20 posts/view = 5B post fetches/day
Average read QPS = 5B / 86,400 ≈ 58,000 QPS
Peak QPS = 58,000 * 3 (typical peak multiplier) ≈ 175,000 QPS
```
At 175K read QPS, a single database server is insufficient. This number immediately justifies an aggressive caching layer and likely read replicas or sharding.

**What interviewers look for**: Getting the right order of magnitude, not the exact number. Noting that this drives architectural decisions. Mentioning peak vs. average.

---

**Q2. You need to store 5 years of user activity logs for 100M users, each generating 10 events per day at approximately 500 bytes per event. What storage capacity do you need?**

Expected approach and answer:
```
Events/day = 100M users * 10 events = 1B events/day
Storage/day = 1B * 500 bytes = 500 GB/day
Storage/year = 500 GB * 365 ≈ 180 TB/year
5-year total = 180 TB * 5 = 900 TB ≈ ~1 PB
With 3x replication factor = ~3 PB total disk
```
This is object storage or a distributed database territory (S3, Cassandra, BigQuery), not a single relational database. The calculation immediately tells you what class of storage you need.

**What interviewers look for**: Applying a replication factor. Noting that this implies a specific storage tier (not standard RDBMS). Estimating cost order-of-magnitude if prompted.

---

**Q3. Your service needs to sustain 50K write QPS to a PostgreSQL database. Each write is a simple INSERT. Can a single PostgreSQL instance handle this? What are your options?**

Expected approach and answer:
```
Typical PostgreSQL inserts: ~10,000-20,000 TPS (simple, with indexes)
50K TPS is 2.5-5x beyond a single instance's typical ceiling.
```
Options, in order of escalating complexity:
1. **Batch inserts**: Buffer writes in the application and bulk-insert in batches of 100-1000. A 100x batch can reduce database round-trips by 100x. Often closes the gap without architectural changes.
2. **Disable synchronous_commit**: For non-critical writes (analytics, logs), async commit dramatically increases throughput at the cost of losing the last ~100ms of writes on crash.
3. **Multiple writer instances with application-level sharding**: Hash the partition key and route writes to the correct instance. Cross-shard queries are now expensive.
4. **Kafka buffer → batch writer**: Accept writes into Kafka, have a batch writer consume and bulk-insert. Decouples write bursts from database throughput. Introduces latency.
5. **Switch to Cassandra or DynamoDB**: Both are designed for high write throughput with LSM trees. Accept the loss of SQL expressiveness.

---

### Bottleneck Identification Questions

**Q4. You have a web service running behind a load balancer with 10 app servers and a PostgreSQL primary with 3 read replicas. Traffic is 20K QPS. Response time p99 is 800ms and climbing. Where do you look first?**

Systematic investigation:
1. **Identify which tier is slow**: Check app server CPU and memory. Check database CPU and I/O. Check connection pool wait time. Look at query latency separately from app processing time.
2. **If database is the bottleneck**: Check `pg_stat_activity` for long-running queries. Check index usage (`pg_stat_user_indexes`). Check if replicas are being used -- are reads actually routing to replicas?
3. **If app servers are the bottleneck**: Check if they are CPU-bound (compute intensive) or I/O-bound (waiting on external calls). Add more app servers if CPU-bound. Identify blocking I/O if I/O-bound.
4. **If connection pool is the bottleneck**: Metrics will show `connection_wait_time > 0`. Install PgBouncer or reduce pool size per instance.

**The key insight**: Do not assume you know where the bottleneck is. Instrument first. The most common mistake is adding app servers when the database is the actual constraint.

---

**Q5. After deploying new code, your cache hit rate drops from 95% to 30% and remains low for 20 minutes before recovering. What happened and how do you prevent it?**

What happened: Cold cache after deploy. If you deploy all instances simultaneously and the old cache is cleared (different cache key format, flush on deploy, or rolling restart that doesn't preserve in-process cache), the new instances start with an empty cache. The 20-minute recovery time is how long it takes to warm to steady-state hit rate.

Prevention options:
1. **Keep the same cache key format across deploys** (most important).
2. **Blue-green with cache warm-up**: Deploy the new version, route a small fraction of traffic to it, wait for the cache to warm, then cut over.
3. **Explicit cache warming**: Before routing traffic, send a batch of synthetic requests representing your hot key distribution.
4. **Stale-while-revalidate**: Serve stale cache entries while revalidating in the background. Cache never fully empties.
5. **In-process cache + distributed cache layering**: In-process caches survive a rolling restart if not all instances restart simultaneously.

---

**Q6. A Kafka consumer group is falling behind. Lag is growing at 10,000 messages/second. The topic has 12 partitions. The consumer group has 6 consumers. Each consumer takes 50ms per message. What is the maximum throughput and what changes fix the lag?**

Expected calculation:
```
Messages each consumer handles per second = 1000ms / 50ms = 20 messages/second
6 consumers * 20 msg/sec = 120 messages/second total
Kafka is producing at rate such that lag grows at 10K/sec
Needed throughput ≥ production_rate (which is production_rate + 10K = current_consumption + 10K)
```

To fix:
1. **Add consumers up to partition count**: Currently 6 consumers for 12 partitions. Add 6 more consumers. Each partition gets its own consumer. Max throughput doubles: 240 messages/second.
2. **Increase partitions**: If 12 consumers at 20 msg/sec (240 total) is still insufficient, increase partition count to 24 and add 24 consumers.
3. **Reduce processing time per message**: 50ms/message is high. Profile the consumer: is it doing synchronous database writes? Batch the writes instead.
4. **Parallelize within a consumer**: If message order within a partition is not required, process messages concurrently within each consumer.

**The constraint**: You can never have more active consumers per group than partitions. Extra consumers sit idle.

---

### Design Decision Questions

**Q7. You are designing the write path for a high-traffic event tracking system (click events, page views, impressions). At peak, you expect 500K events/second. The data must be available for analytics queries within 60 seconds. How do you design the ingestion layer?**

Key constraints: 500K writes/second, 60-second latency requirement.

Design:
```
Clients → API Gateway → Kafka (event bus) → Stream Processor → Analytics Store
```

Why each choice:
- **API Gateway**: Rate limiting, authentication, batching of client events.
- **Kafka**: Accept 500K/sec with low latency. Append-only log is extremely fast. Durable. Consumer groups allow multiple downstream consumers.
- **Stream Processor (Flink/Kafka Streams)**: Aggregate events in real time (count by event type, by user, by page). Emit micro-batch writes to analytics store every 10-30 seconds.
- **Analytics Store (ClickHouse, BigQuery, or Druid)**: Columnar storage optimized for aggregate reads. Batch ingest from stream processor satisfies 60-second SLA.

Trade-offs to call out:
- Kafka partitioning strategy: partition by event type for ordering guarantees, or by user for user-level sequencing. Not both.
- At-least-once delivery means duplicates. Analytics store must deduplicate or tolerate double-counting.
- 60-second SLA is easily achieved; this design typically lands at 10-20 seconds. If SLA was 1 second, we would need a different analytics store or a separate hot path.

---

**Q8. You are designing a rate limiter for an API that allows 1,000 requests per user per minute. The API is served by 50 application servers. Choose an algorithm and explain how you handle the distributed case.**

Algorithm options:
- **Fixed window counter**: Simplest. Count requests in 1-minute windows. Allows 2x limit at window boundary (spike at end of one window + start of next).
- **Sliding window log**: Store timestamps of each request. Precise, no boundary problem. Memory intensive at high request volumes.
- **Sliding window counter**: Interpolate between fixed windows. Approximation, but much more memory efficient than log. Industry standard choice.
- **Token bucket**: Tokens refill at a constant rate. Allows bursts up to bucket capacity. Natural and intuitive. Slightly more complex to implement distributedly.

Distributed implementation for sliding window counter:
```
// Redis script (atomic):
key = "rate_limit:{user_id}:{current_minute_bucket}"
prev_key = "rate_limit:{user_id}:{previous_minute_bucket}"

current_count = INCR key
EXPIRE key 120  // 2 minutes for safety

// Sliding interpolation:
current_window_elapsed = (current_timestamp % 60_seconds) / 60
prev_count = GET prev_key OR 0
weighted = (prev_count * (1 - current_window_elapsed)) + current_count
ALLOW if weighted <= 1000
```

Key design decisions to call out:
- **Centralized Redis** vs. **local + gossip**: Centralized Redis is authoritative but adds latency per request. Local counters with periodic sync to Redis are faster but allow brief overages.
- **Lua scripts for atomicity**: The check-then-increment operation must be atomic. Use Redis Lua scripts or the INCR + EXPIRE pattern.
- **What to do when Redis is down**: Fail open (allow traffic, no rate limiting) vs. fail closed (block all traffic). For most APIs, fail open is the right default.

---

**Q9. You are building a notification system that sends push notifications, emails, and SMS. At peak, you need to send 10M notifications in under 5 minutes. How do you design this?**

Estimation:
```
10M notifications in 5 minutes = 10M / 300 seconds ≈ 33,000 notifications/second
```

At 33K notifications/second, the bottleneck is throughput to external providers (FCM, Sendgrid, Twilio), not your internal systems.

Design:
```
Event Source → Notification Queue (Kafka) → Dispatcher → Provider Queue (per type)
                                                        → Push notifications (Kafka partition → FCM workers)
                                                        → Email (Kafka partition → Sendgrid workers)
                                                        → SMS (Kafka partition → Twilio workers)
```

Key decisions:
1. **Prioritization**: Not all 10M notifications are equal urgency. Segment by priority (transactional vs. marketing). Transactional (password reset, purchase confirmation) gets a separate high-priority queue. Marketing batch can tolerate delay.
2. **Provider rate limits**: FCM handles ~1M/second, Sendgrid ~1000 emails/second (varies by tier), Twilio ~1000 SMS/second. Design worker pools within provider limits.
3. **Fan-out strategy**: A single triggered event (e.g., a flash sale) may generate 10M notifications. The event should be published once; the dispatcher handles fan-out to individual users. Never fan-out in the event source.
4. **Retry and dead letter**: Provider failures are common. Exponential backoff with DLQ. Distinguish retriable errors (rate limit, timeout) from permanent failures (invalid token, invalid number).
5. **Idempotency**: At-least-once delivery from Kafka means the notification sender may receive the same message twice. Deduplicate at send time using a `notification_id` written to Redis with TTL.

---

### Failure Scenario Questions

**Q10. Your primary database goes down unexpectedly. You have 3 read replicas. What happens to your system, what is your response, and what could you have done in the design to reduce impact?**

Immediate impact:
- All writes fail immediately.
- Reads may continue from replicas if your application is designed to route reads to replicas. If reads only go to the primary, all reads also fail.
- Background jobs, batch processes, and anything requiring a write (including session creation) fails.

Response:
1. Alert fires (you have a healthcheck on the primary, right?).
2. Promote a replica to primary: `pg_promote()` in PostgreSQL 12+. This takes 30 seconds to minutes depending on replication lag.
3. Update application configuration to point to the new primary (service discovery or config change).
4. Monitor replica lag on promotion: if the promoted replica was 30 seconds behind, 30 seconds of writes are lost unless you have synchronous replication.

Design mitigations:
- **Automatic failover**: Use a managed database service (AWS RDS Multi-AZ, Cloud SQL HA) with automatic promotion. Reduces failover time from minutes to ~30-60 seconds.
- **Synchronous replication to at least one replica**: Guarantees zero data loss at the cost of write latency (+round-trip time to the replica).
- **Health checks with circuit breakers**: Detect primary failure at the application layer and fail fast with 503s rather than allowing requests to queue.
- **Read path survives independently**: Route reads to replicas always, not just on primary failure. This means read traffic is unaffected by primary downtime.

---

**Q11. Your caching layer (Redis) goes down entirely. Your system has 50K QPS, 90% of which would have been served from cache. What happens and how should you have designed for this?**

Immediate impact:
- 45K QPS of cache hits now fall through to the database.
- Your database is sized for 5K QPS (the 10% that was cache misses). It immediately becomes overwhelmed.
- Database latency climbs, connections exhaust, and the entire system degrades or falls over.

This is the classic "thundering herd on cache failure" scenario. The cache failure cascades into a database failure.

Design to prevent cascade:
1. **Redis Cluster or Redis Sentinel**: Single Redis node is a single point of failure. Redis Cluster shards across nodes. Redis Sentinel provides automatic failover for a single primary.
2. **Circuit breaker on cache**: Detect Redis failure quickly. When Redis is down, apply request rate limiting to protect the database -- accept reduced throughput rather than database failure.
3. **Database capacity headroom**: Size the database to handle some cache miss surge. A database that can handle only exact steady-state cache-miss traffic has no failure headroom.
4. **Fallback tier**: If Redis is unavailable, fall back to in-process local cache (limited size, no consistency) rather than directly to the database.
5. **Gradual recovery**: When Redis comes back up, do not immediately send all traffic through it. Warm the cache before cutting over to prevent a second stampede.

---

**Q12. You deploy a new version of a service. Within 2 minutes, p99 latency on the dependent service spikes from 100ms to 8 seconds. No alerts fired on the new service itself. What do you investigate?**

Pattern: The new service is not slow itself, but it is making the downstream service slow. Classic signs of a missing connection pool limit or unbounded concurrency.

Investigation path:
1. **Check the new service's outbound calls**: Is it calling the dependent service synchronously in a loop (N+1 regression)? Did the new version remove a cache layer, causing every request to hit the downstream service?
2. **Check connection counts on the downstream service**: Is the new service opening significantly more connections than the old version? A pool size configuration change or removal of connection limits could cause connection exhaustion on the downstream.
3. **Check request rate**: Did the new service introduce retry logic that is amplifying requests? A retry storm from the upstream can cause 5-10x the actual request rate on the downstream.
4. **Rollback if source is confirmed**: Once you identify the new service as the cause, roll back immediately. Investigate in staging.

Common root causes:
- Removed connection pooling (added a new database client without pool configuration).
- Added synchronous loop over a downstream call (N+1).
- Added aggressive retry with no exponential backoff (retry storm).
- Removed a caching layer, dramatically increasing downstream call rate.

---

### Code-Level Scaling Questions

**Q13. You have this Go code that loads a user's orders. It works fine in staging with 100 users. It's destroying the database in production with 100K users. What's wrong and how do you fix it?**

```go
func GetUserDashboard(userID int) (*Dashboard, error) {
    user, err := db.GetUser(userID)
    if err != nil {
        return nil, err
    }

    orders, err := db.GetOrdersByUser(userID)
    if err != nil {
        return nil, err
    }

    for i, order := range orders {
        product, err := db.GetProduct(order.ProductID)
        if err != nil {
            return nil, err
        }
        orders[i].Product = product
    }

    return &Dashboard{User: user, Orders: orders}, nil
}
```

**Problem**: N+1 query. If a user has 50 orders, this makes 52 database queries per request: 1 for user, 1 for orders, 50 for products. At 1K concurrent requests per second with users averaging 50 orders: 52,000 database queries per second from a single endpoint.

**Fix: batch the product lookup**:
```go
func GetUserDashboard(userID int) (*Dashboard, error) {
    user, err := db.GetUser(userID)
    if err != nil {
        return nil, err
    }

    orders, err := db.GetOrdersByUser(userID)
    if err != nil {
        return nil, err
    }

    // Collect all product IDs
    productIDs := make([]int, len(orders))
    for i, order := range orders {
        productIDs[i] = order.ProductID
    }

    // Single batch query: SELECT * FROM products WHERE id IN (...)
    products, err := db.GetProductsByIDs(productIDs)
    if err != nil {
        return nil, err
    }

    // Build index for O(1) lookup
    productMap := make(map[int]*Product, len(products))
    for _, p := range products {
        productMap[p.ID] = p
    }

    for i, order := range orders {
        orders[i].Product = productMap[order.ProductID]
    }

    return &Dashboard{User: user, Orders: orders}, nil
}
```

Result: 3 database queries regardless of order count.

---

**Q14. This endpoint is seeing a traffic spike. 10,000 requests per second are all hitting the database. The cache miss rate is 100% for 30 seconds after the spike starts, then stabilizes at 5%. What is likely happening and how do you fix it in code?**

```go
func GetProductPrice(productID int) (float64, error) {
    cached, err := redis.Get(fmt.Sprintf("price:%d", productID))
    if err == nil {
        return strconv.ParseFloat(cached, 64)
    }

    price, err := db.GetPrice(productID)
    if err != nil {
        return 0, err
    }

    redis.Set(fmt.Sprintf("price:%d", productID), price, 0) // No TTL!
    return price, nil
}
```

**Problem**: `TTL = 0` means no expiration. The cache fills with entries that never expire. When a traffic spike occurs, the database is fine because the cache has everything. But when the code was first deployed (or Redis was flushed), there was a 30-second stampede until the cache warmed up.

The 30-second recovery time is how long it takes to cache enough of the hot key space.

Secondary problem: No TTL means prices never refresh. Price updates in the database do not reflect in the API until Redis is explicitly cleared.

**Fix**:
```go
const (
    priceCacheTTL    = 5 * time.Minute
    priceCacheTTLMax = 6 * time.Minute // jitter range
)

func GetProductPrice(productID int) (float64, error) {
    key := fmt.Sprintf("price:%d", productID)

    cached, err := redis.Get(key)
    if err == nil {
        return strconv.ParseFloat(cached, 64)
    }

    // Distributed lock to prevent stampede
    lock, err := redis.SetNX(key+":lock", "1", 5*time.Second)
    if !lock || err != nil {
        // Another goroutine is fetching -- wait briefly and retry from cache
        time.Sleep(50 * time.Millisecond)
        return GetProductPrice(productID)
    }
    defer redis.Del(key + ":lock")

    price, err := db.GetPrice(productID)
    if err != nil {
        return 0, err
    }

    // Jittered TTL to prevent synchronized expiration
    ttl := priceCacheTTL + time.Duration(rand.Int63n(int64(priceCacheTTLMax-priceCacheTTL)))
    redis.Set(key, price, ttl)

    return price, nil
}
```

---

**Q15. You are reviewing a pull request for a Kafka consumer. The reviewer says it will not scale to 100K messages/second. What is wrong?**

```go
func ConsumeMessages(consumer *kafka.Consumer) {
    for {
        msg, err := consumer.ReadMessage(10 * time.Second)
        if err != nil {
            log.Printf("Error: %v", err)
            continue
        }

        result, err := processMessage(msg)
        if err != nil {
            log.Printf("Failed to process message %s: %v", msg.Key, err)
            continue // silently drop and move on
        }

        err = db.SaveResult(result)
        if err != nil {
            log.Printf("Failed to save: %v", err)
            continue // silently drop
        }

        consumer.CommitMessage(msg) // commit after each message
    }
}
```

**Problems**:

1. **Single-threaded**: One goroutine processes one message at a time. At 100K msg/sec, each message has a budget of 10 microseconds. Any I/O (database write) takes milliseconds. This consumer will fall behind immediately.

2. **Commit after every message**: Auto-commit at the message level is the slowest possible commit strategy. Batch commits amortize the overhead.

3. **Silent message drops on error**: Processing failures and database failures cause messages to be silently dropped. There is no retry, no DLQ, and no visibility into failure rate.

4. **No backpressure**: If the database is slow, the consumer keeps reading messages into memory with nowhere to send them.

**Better design**:
```go
func ConsumeMessages(consumer *kafka.Consumer, concurrency int) {
    work := make(chan *kafka.Message, concurrency*10)
    var wg sync.WaitGroup

    // Worker pool
    for i := 0; i < concurrency; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for msg := range work {
                if err := processAndSave(msg); err != nil {
                    sendToDLQ(msg, err) // don't drop -- route to DLQ
                }
            }
        }()
    }

    // Batch commit
    var pending []*kafka.Message
    for {
        msg, err := consumer.ReadMessage(10 * time.Second)
        if err != nil {
            continue
        }
        work <- msg
        pending = append(pending, msg)

        if len(pending) >= 1000 {
            consumer.CommitOffsets() // commit batch of 1000
            pending = pending[:0]
        }
    }
}
```

---

## Related Reading

- [Module 01: System Design Framework Essentials](01-system-design-framework-essentials.md) — the framework structure this module builds on top of, especially the estimation and deep dive sections
- [Module 01: Advanced System Design](02-advanced-system-design.md) — staff-level extensions including multi-region design and organizational considerations
- [Module 02: Indexing, Sharding, and Replication](../02-databases-at-scale/02-indexing-sharding-and-replication.md) — deep coverage of consistent hashing, cross-shard queries, and the sharding decisions referenced in this module
- [Module 03: Advanced Caching Systems](../03-caching/03-advanced-caching-systems.md) — cache stampede prevention mechanisms including probabilistic early rehydration and distributed locking
- [Module 04: Message Queue Operations and Patterns](../04-message-queues/03-message-queue-operations-and-patterns.md) — Kafka internals, consumer group design, and the outbox pattern referenced in the Kafka consumer question
- [Module 10: Classic Design Problems](../10-classic-problems/01-classic-design-problems.md) — applies the scaling mental model to end-to-end design problems: fan-out in news feeds, rate limiting, and notification systems

---

## Key Takeaways

1. **Lead with failure modes.** The single clearest signal of production experience is identifying what breaks before the interviewer asks. State the failure mode, then the mitigation.

2. **Use the narrative pattern.** Before state → inflection point → decision with trade-offs → after state. Even hypothetical answers are more credible in this structure.

3. **Estimation drives architecture.** Calculate QPS and storage before designing. The numbers tell you which stage of the scaling progression you are in and prevent over-engineering.

4. **Scale is not linear.** Hot partitions, thundering herds, and N+1 regressions are scale problems, not correctness problems -- they pass all tests and destroy production. Learn to recognize them by pattern.

5. **The scaling progression is your anchor.** Start simple, identify the first bottleneck, evolve to the next stage. Do not jump to Stage 6 for a Stage 2 problem.

6. **Latency percentiles, not averages.** Design against p99. State non-functional requirements as "p99 < 200ms" not "low latency."
