# Stripe TSE Onsite Prep — March 25, 2026

## Your Interview

| Time (PDT) | Round | Interviewer | Duration |
|---|---|---|---|
| 11:00–11:45 | Programming Exercise | Nicholas Xavier | 45 min |
| 11:45–12:45 | Experience & Goals | Ali Riaz | 60 min |
| 1:30–2:15 | Users First & Curious | Tyler Martin | 45 min |
| 2:30–3:30 | Integration | Adam Fuller | 60 min |

All on Zoom. Same link between rounds. AI tools strictly prohibited.

## Course Structure

### Knowledge (read these)

| File | What | Priority |
|---|---|---|
| `05-onsite-guide.md` | **Start here.** Round-by-round strategy, prep plan, day-of checklist | Read first |
| `01-api-fundamentals-and-core-objects.md` | Stripe API design, auth, idempotency, webhooks | High |
| `02-payments-deep-dive.md` | PaymentIntents, refunds, disputes | High |
| `03-billing-and-subscriptions.md` | Subscriptions, invoices, proration | Medium |
| `04-connect-and-platform-payments.md` | Connect, multi-party payments | Medium |

### Drills (code these)

Run with `cd drills && make d1` or `npx tsx drills/drill_01_rate_limiter.ts`.

| Drill | Pattern | Target | Preps for |
|---|---|---|---|
| `toolbox.ts` | Copy-paste patterns: Maps, sorting, HTTP, union-find | Review | All rounds |
| `drill_01_rate_limiter.ts` | Fixed/sliding window, token bucket | 30 min | Programming |
| `drill_02_record_dedup.ts` | Exact/fuzzy match, transitive merge | 30 min | Programming |
| `drill_03_transaction_ledger.ts` | Double-entry, history, batch + rollback | 30 min | Programming |
| `drill_04_log_sanitizer.ts` | Regex redaction, streaming | 25 min | Programming |
| `drill_05_api_client.ts` | CRUD, pagination, retry, webhooks | 30 min | Integration |
| `drill_06_express_api.ts` | Routes, middleware, Stripe API, testing | 35 min | Integration |
| `drill_07_tiered_pricing.ts` | Flat/tiered/base+overflow pricing, multi-product | 30 min | Programming |
| `drill_08_currency_conversion.ts` | Graph BFS/DFS, multi-hop rates, best path | 30 min | Programming |
| `drill_09_string_parsing.ts` | Accept-Language, q-values, invoice reconciliation | 30 min | Programming |
| `drill_10_event_scheduler.ts` | Subscription notifications, plan changes, store closing penalty | 30 min | Programming |

### Simulations (timed practice)

| Directory | Simulates | Timer | Notes |
|---|---|---|---|
| `projects/01-programming-sim/` | Programming Exercise (Nicholas Xavier) | 40 min | Payment fee calculator — tiered pricing, validation, settlement |
| `projects/02-experience-and-goals/` | Experience & Goals (Ali Riaz) | — | Project deep dives, "Why Stripe?", interviewer questions |
| `projects/03-users-first-and-curious/` | Users First & Curious (Tyler Martin) | — | User empathy + curiosity story bank, 6 STAR prompts |
| `projects/04-integration/` | Integration (Adam Fuller) | 50 min | Add payments/refunds/webhooks to existing codebase |

### Reference

| File | What |
|---|---|
| `docs/TSE_Prep_Document.pdf` | Official Stripe interview prep document |
| `docs/What to Expect_ Video Interviews.pdf` | Official Stripe video interview guide |
