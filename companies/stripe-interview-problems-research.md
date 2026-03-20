# Stripe Interview Problems: Comprehensive Research from Candidate Reports

> Research compiled from Reddit, Blind, Glassdoor, LeetCode discussions, Medium, interviewing.io,
> 1Point3Acres, programhelp.net, linkjob.ai, staffengprep.com, and other candidate reports.
> Last updated: 2026-03-19

---

## Table of Contents

1. [Programming Exercise Problems (Phone Screen / Onsite Coding)](#programming-exercise-problems)
2. [Online Assessment (HackerRank OA) Problems](#online-assessment-problems)
3. [Integration Exercise Problems](#integration-exercise-problems)
4. [Bug Squash / Debug Round Problems](#bug-squash--debug-round-problems)
5. [stripe-interview GitHub Organization](#stripe-interview-github-organization)
6. [Problem Index by Pattern](#problem-index-by-pattern)

---

## Programming Exercise Problems

These are the problems used in phone screens and onsite coding rounds. They are typically
1 problem with 3-4 progressive parts, each building on the previous. You must solve Part 1
to unlock Part 2. The total time is 60-90 minutes.

---

### 1. Shipping Cost Calculator (3 parts)

**Type:** Programming Exercise (Phone Screen / Onsite)
**Reported on:** Glassdoor, 1Point3Acres, linkjob.ai, staffengprep.com

**Problem:** Implement a shipping-cost calculator with progressively richer pricing models
based on a configuration object keyed by destination/product.

**Part 1 -- Flat per-unit pricing:**
- Given a configuration with a `unitPrice`, return `unitPrice * quantity`.
- Straightforward multiplication.

**Part 2 -- Tiered per-unit pricing:**
- Given a configuration with `tiers` (each tier has an `upTo` quantity and `unitPrice`),
  compute the total by applying each tier only to the items that fall within that tier's range.
- Each `upTo` is inclusive; accumulate across tiers.
- Example: tiers [{upTo: 5, unitPrice: 10}, {upTo: 10, unitPrice: 8}, {upTo: Infinity, unitPrice: 5}]
  - For quantity=7: first 5 at $10 = $50, next 2 at $8 = $16, total = $66

**Part 3 -- Flat base with tiered overflow:**
- Given a configuration with a `baseFlat` (charge F once for quantities 1..n) plus tiers afterward.
- For quantities > n, apply tiered pricing to the overflow.

**Patterns/Data Structures:**
- Hash maps for configuration lookup
- Careful off-by-one boundary handling
- Modular function design (each pricing model should be its own function)

---

### 2. Transaction Balance / Account Balancing (3 parts)

**Type:** Programming Exercise (Phone Screen / OA / Onsite)
**Reported on:** Glassdoor, LeetCode, linkjob.ai, codinginterview.com

**Problem:** Given an array of transactions where each transaction is
`[from_id, to_id, amount]` meaning person `from_id` gave `$amount` to person `to_id`.

**Part 1 -- Compute non-zero balances:**
- Output the names/IDs of all users with a non-zero balance and their corresponding balances.
- Requires iterating through transactions, maintaining a balance map.

**Part 2 -- Reject invalid transactions:**
- Output a `rejected_transactions` list: reject any transaction where a user's
  `current_balance + amount` would go negative.
- Process transactions in order; rejected ones don't affect balances.

**Part 3 -- Borrowing mechanism:**
- Implement a borrowing/lending feature where accounts can borrow from the platform
  to avoid rejection. Track borrowed amounts separately.

**Patterns/Data Structures:**
- Hash map (user -> balance)
- Sequential processing with state
- Edge cases: zero amounts, self-transfers, ordering

---

### 3. Email Subscription / Notification Scheduler (3 parts)

**Type:** Programming Exercise (Phone Screen / Onsite)
**Reported on:** linkjob.ai, Glassdoor, 1Point3Acres

**Problem:** Given a list of users `{name, plan, begin_date, duration}`, implement an email
notification scheduling system.

**Part 1 -- Basic scheduling:**
- Send welcome email on plan start date
- Send upcoming expiration email 15 days before expiration
- Send expiration email on expiration date
- Output format: `"<day>: [<Email Type>] Subscription for <name> (<plan>)"`
- Notifications ordered by time; ties broken by subscription ID

**Part 2 -- Plan changes:**
- Add a plan changes list `{name, new_plan, change_date}`
- Output `[Changed]` messages
- Recalculate remaining duration and reschedule future notifications based on new timeline

**Part 3 (Bonus) -- Renewals:**
- Add renewal feature with extension data in change list `{name, extension, change_date}`
- Print `[Renewed]` messages
- Reschedule expiry notifications

**Patterns/Data Structures:**
- Priority queue or sorted event list
- Hash maps for user state
- Date/time arithmetic
- Event-driven simulation

---

### 4. Currency Conversion (3 parts)

**Type:** Programming Exercise (Phone Screen / Onsite)
**Reported on:** staffengprep.com, linkjob.ai, bigtechexperts.com, interviewing.io

**Problem:** Given exchange rates as string `"AUD:USD:0.7,AUD:JPY:100,USD:CAD:1.2"`,
implement currency conversion.

**Part 1 (~10 min) -- Direct conversion:**
- Build bidirectional map of exchange rates
- Convert between two currencies with a direct rate, or return -1 if unavailable
- Remember: if USD:CAD:1.3, then CAD:USD = 1/1.3
- Example: convert 100 USD to CAD -> 130.0

**Part 2 (~20 min) -- Multi-hop conversion (any path):**
- Support conversions through intermediate currencies
- Return route, effective rate, and converted amount
- Example: USD->CAD->AUD, rate=1.76, value=176.0
- Use DFS to find any valid conversion path

**Part 3 (~25 min) -- Shortest conversion path:**
- Find the path requiring fewest intermediate conversions
- BFS guarantees minimum hops
- Track parents and accumulate rates along path

**Patterns/Data Structures:**
- Graph (adjacency list)
- BFS for shortest path
- DFS for any path
- Hash map for exchange rate storage

---

### 5. Accept-Language Header Parser (4 parts)

**Type:** Programming Exercise (Phone Screen)
**Reported on:** Glassdoor, staffengprep.com, codinginterview.com

**Problem:** Parse an HTTP `Accept-Language` header and return suitable languages from a
set of supported languages, in order of preference.

Example header: `Accept-Language: en-US, fr-CA, fr-FR`

**Part 1 -- Basic matching:**
- Return matching languages from supported set in order of preference
- Simple string parsing and lookup

**Part 2 -- Language variants:**
- Support non-region-specific tags (e.g., "en" matches "en-US", "en-GB")
- A generic tag matches all specific variants of that language

**Part 3 -- Wildcard support:**
- Support `*` wildcard meaning "all other languages"
- Languages explicitly listed take precedence over wildcard

**Part 4 -- Quality factors (q-values):**
- Parse explicit numeric weights: `en-US;q=0.8,fr;q=0.9`
- Default q=1.0 if not specified
- Sort by q-value descending, then by original order for ties
- Support q=0 meaning "specifically not wanted"

**Patterns/Data Structures:**
- String parsing (split on commas, semicolons)
- Sorting with custom comparators
- Hash map for language->quality mapping
- Prefix matching for language variants

---

### 6. Account Scheduler / Lock Manager (3 parts)

**Type:** Programming Exercise (Onsite)
**Reported on:** linkjob.ai, 1Point3Acres

**Problem:** Implement an account scheduling system with locking.

**Part 1 -- Check availability:**
- Implement `is_available(account_id, timestamp)` method
- Accounts have `locked_until` timestamps
- Return true if current time >= locked_until

**Part 2 -- Acquire with duration:**
- Implement `acquire(account_id, duration)` to lock an account
- Updates the account's `locked_until` timestamp to `current_time + duration`

**Part 3 -- LRU auto-selection:**
- Extend `acquire` so that if called without a specific account_id, the system
  automatically picks the available account that was least recently used
- Classic LRU cache pattern applied to account management

**Patterns/Data Structures:**
- Hash map (account_id -> locked_until timestamp)
- LRU Cache (OrderedDict or doubly-linked list + hash map)
- Timestamp comparison

---

### 7. User Deduplication / Record Matching (3 parts)

**Type:** Programming Exercise (Phone Screen)
**Reported on:** linkjob.ai

**Problem:** Match user records based on weighted similarity across multiple fields.

**Part 1 -- Pairwise similarity:**
- Calculate similarity scores using individual field weights (name, email, company)
- Identify linked users exceeding a threshold score

**Part 2 -- Transitive links (1-hop):**
- Find indirect connections: if A matches B and B matches C, then A-B-C are linked
- 1-hop transitive linking

**Part 3 -- Connected components (unlimited hops):**
- Discover entire connected components with unlimited hop distance
- All users transitively connected form one group

**Patterns/Data Structures:**
- String similarity metrics
- Union-Find (Disjoint Set Union) for connected components
- Graph traversal (BFS/DFS)
- Hash maps for field comparison

---

### 8. Store Closing Time Penalty (3 parts)

**Type:** Programming Exercise (Phone Screen / OA)
**Reported on:** LeetCode, Glassdoor, programhelp.net, codinginterview.com
**LeetCode equivalent:** 2483. Minimum Penalty for a Shop

**Problem:** Given a string of 'Y' (customer arrives) and 'N' (no customer) representing
hourly logs of customer visits.

**Part 1 -- Compute penalty for given closing time:**
- Penalty = +1 for each hour open with no customers ('N' before closing)
             +1 for each customer missed ('Y' after closing)
- Input: log string and closing time integer

**Part 2 -- Find best closing time:**
- Return the earliest hour to close (0 to n inclusive) minimizing total penalty
- If tied, return smallest value
- Optimal: O(n) using running penalty update

**Part 3 -- Aggregate multiple logs:**
- Parse logs with `BEGIN`/`END` markers
- Extract valid sequences, ignore nesting and garbage text
- Return best closing times for each valid log

**Patterns/Data Structures:**
- Prefix sum / running counter
- Greedy optimization
- String parsing for log extraction
- Stack or state machine for BEGIN/END parsing

---

### 9. Credit Card Number Masking / Redaction (3 parts)

**Type:** Programming Exercise (Phone Screen / Onsite)
**Reported on:** interviewing.io, staffengprep.com, codinginterview.com

**Problem:** Blur/mask credit card numbers from log strings.

**Part 1 -- Basic masking:**
- Replace digits with 'X' except last 4 digits for tokens containing 13-16 digits
- Use regex: `r'\b(\d{12})(\d{4})\b'`

**Part 2 -- Brand validation:**
- Detect card network:
  - VISA: 13 or 16 digits starting with 4
  - MASTERCARD: 16 digits starting with 51-55
  - AMEX: 15 digits starting with 34 or 37
- Output network name or "UNKNOWN"

**Part 3 -- Luhn checksum validation:**
- Implement Luhn algorithm validation alongside brand rules
- Double every second digit from right, subtract 9 if > 9, sum all, check mod 10 == 0
- Return network name if valid, "INVALID_CHECKSUM" if fails, "UNKNOWN_NETWORK" if no match

**Patterns/Data Structures:**
- Regular expressions
- String manipulation
- Luhn algorithm implementation
- Conditional branching based on prefix/length

---

### 10. Invoice / Payment Reconciliation (multi-part)

**Type:** Programming Exercise (Onsite Coding)
**Reported on:** LeetCode, 1Point3Acres, interviewexperiences.in

**Problem:** Stripe's Invoicing product -- match incoming payments to invoices.

**Input format:**
- Payment: `"paymentID,amount,Paying off: invoiceID"`
- Invoices: `["invoiceA,2024-01-01,100", "invoiceB,2024-02-01,200"]`

**Output:** `"payment5 pays off 1000 for invoiceC due on 2023-01-30"`

**Key requirements:**
- Parse comma-delimited strings
- Extract invoice ID from memo line
- Match payment to invoice
- Format output string with reconciliation details

**Follow-ups may include:**
- Partial payments (payment amount < invoice amount)
- Multiple payments against one invoice
- Overpayments and refund calculation

**Patterns/Data Structures:**
- String parsing (split on commas, extract from patterns)
- Hash map (invoice_id -> invoice details)
- Formatting output strings

---

### 11. Rate Limiter (multi-part)

**Type:** Programming Exercise (Phone Screen / Onsite)
**Reported on:** interviewing.io, prepfully.com, codinginterview.com, Glassdoor

**Problem:** Design a rate limiter that handles 100 requests per minute per user.

**Core implementation:**
- Maintain a sliding window of timestamps per user
- Remove entries older than the window
- Check if count exceeds limit before allowing request

**Stripe's actual rate limiting approaches (from Paul Tarjan's gist):**
1. **Token Bucket:** Replenish rate + burst capacity, atomic Redis operations
2. **Concurrent Requests Limiter:** Track active requests per user in sorted set
3. **Fleet Usage Load Shedder:** Global concurrent request limiting
4. **Worker Utilization Load Shedder:** Probabilistic dropping based on utilization

**Patterns/Data Structures:**
- Hash map + queue/deque per user
- Sliding window
- Token bucket algorithm
- Timestamp comparison

---

### 12. Account Balance Transfer System (multi-part)

**Type:** Programming Exercise (Onsite)
**Reported on:** programhelp.net, linkjob.ai, LeetCode
**LeetCode equivalent:** 465. Optimal Account Balancing

**Problem:** Implement a transfer system that balances all accounts to zero.

**Part 1:** Calculate positive/negative account balances from transaction list
**Part 2:** Offset balances through transfers; focus on correctness
**Follow-up:** Minimize transaction count using greedy matching or DFS with pruning
**Follow-up:** Design audit logging with validation and traceability

**Patterns/Data Structures:**
- Hash map for balances
- Greedy matching (pair largest debtor with largest creditor)
- DFS with backtracking for optimal solution
- Sorting

---

### 13. Chargeback / Dispute Parser (3 parts)

**Type:** Programming Exercise (OA / Coding)
**Reported on:** programhelp.net

**Problem:** Parse refund dispute data from banks so disputes can be presented to merchants.

**Part 1 -- Parse valid refund data:**
- Extract refund information from structured data
- Output summarized dispute details in human-readable format
- All inputs valid in this section

**Part 2 -- Filter invalid/corrupted data:**
- Remove entries that cannot be properly parsed
- Handle corrupted network data gracefully

**Part 3 -- Filter withdrawn disputes:**
- Exclude refund requests that users withdrew after submission
- When same transaction ID appears in multiple rows and a later entry shows "withdrawn",
  exclude both from output

**Patterns/Data Structures:**
- String parsing
- Data validation / error handling
- Hash map for tracking transaction IDs and their status
- Filtering logic

---

### 14. Brace Expansion / String Expansion

**Type:** Programming Exercise (Phone Screen)
**Reported on:** codinginterview.com, LeetCode
**LeetCode equivalent:** 1087. Brace Expansion

**Problem:** Given a pattern string like `{a,b}c{d,e}f`, generate all possible words
in lexicographical order.

**Output:** `['acdf', 'acef', 'bcdf', 'bcef']`

**Approach:**
- Parse string into groups (single chars or sets from braces)
- Sort each set
- Compute Cartesian product of all groups
- Return sorted results

**Patterns/Data Structures:**
- String parsing with stack or state machine
- Cartesian product (itertools.product in Python)
- Lexicographic sorting

---

### 15. CSV Data Processing / File I/O (3 parts)

**Type:** Programming Exercise (Coding)
**Reported on:** linkjob.ai

**Part 1 -- Read and aggregate:**
- Read CSV file of transactions
- Filter by status
- Output total per user

**Part 2 -- API integration with error handling:**
- Make HTTP request to mock payment API
- Handle errors and timeouts
- Parse response

**Part 3 -- Code refactoring and unit testing:**
- Refactor messy codebase for readability
- Write automated tests with mocking

**Patterns/Data Structures:**
- File I/O, CSV parsing
- HTTP requests
- Hash map for aggregation
- Test design patterns

---

### 16. Min/Max Record Finder (4 parts)

**Type:** Programming Exercise (Phone Screen)
**Reported on:** blog.rampatra.com (Dublin interview)

**Part 1:** Write function to fetch the record with the minimum value
**Part 2:** Accept additional parameter, retrieve both min and max based on parameter
**Part 3:** Implement using a comparator pattern
**Part 4:** Handle tie scenarios

Boilerplate code provided. Completed in ~45 minutes.

**Patterns/Data Structures:**
- Comparators / lambda functions
- Linear scan
- Handling ties (return all or first)

---

### 17. Fraud Score Calculation (multi-part)

**Type:** Programming Exercise (OA / Onsite)
**Reported on:** programhelp.net, linkjob.ai

**Problem:** Given merchants (with names and initial scores), transactions
(merchant M, customer C, amount h), and rules (thresholds), calculate final fraud scores.

**Scoring Rules:**
1. Group by (merchant, customer): If pair has >= 3 transactions, add total amount to merchant score
2. Group by (merchant, customer, amount): If combination appears >= 3 times, add total again
3. Output each merchant's name and final score, sorted alphabetically

**Key pitfalls:**
- Boundary cases: empty transactions, single customers, duplicate amounts
- Floating-point precision
- Strict comparison (>) for thresholds

**Patterns/Data Structures:**
- Nested hash maps for grouping
- Counting with conditional aggregation
- Sorting

---

### 18. Tiered Pricing / Accounting System (3 parts)

**Type:** Programming Exercise (Phone Screen -- Intern)
**Reported on:** linkjob.ai (intern interview)

**Part 1:** Calculate total price given an order and shipping cost
**Part 2:** Implement tiered pricing where unit price decreases with quantity increases
**Part 3:** Support two cost calculation methods:
  - Incremental method (from Part 2)
  - Fixed-pricing model where total cost remains consistent within quantity ranges

**Patterns/Data Structures:**
- Tiered calculation logic
- Strategy pattern for different pricing models
- Configuration-driven pricing

---

## Online Assessment Problems

These are HackerRank OA problems, typically 60 minutes for 1 question with 3-4 parts,
or 90 minutes for 2-3 questions. Problems are drawn from a known bank.

---

### OA-1. Atlas Company Name Check (3 parts)

**Reported on:** programhelp.net, linkjob.ai

**Part 1 -- Basic availability check:**
Normalize company names by:
- Ignoring case
- Treating "&" and "," as spaces
- Collapsing consecutive spaces
- Removing standard suffixes (Inc., Corp., LLC and variants)
- Dropping leading "The/An/A"
- Removing "And" unless it starts the name
- Compare normalized name against registered database

**Part 2 -- Persistent registration:**
- Maintain permanent records across all requests
- Prevent resubmission of previously registered names

**Part 3 -- Reclamation requests:**
- Support RECLAIM commands that remove names from unavailable list
- Only original registrant can reclaim their normalized name

---

### OA-2. Card Range Obfuscation (multi-part)

**Reported on:** programhelp.net, linkjob.ai

**Input:** 6-digit BIN (Bank Identification Number) and count of intervals with 10-digit offsets and brand names.

**Task:** Fill gaps in card ranges to ensure complete coverage of the entire BIN range.

**Process:**
- BIN range: Start = BIN * 10^12, End = BIN * 10^12 + 10^12 - 1
- Sort intervals, merge overlapping ones
- Fill gaps between consecutive intervals
- Output gap-free intervals sorted by start, with full 16-digit card numbers

**Example:** BIN `777777` with two intervals produces continuous VISA and MASTERCARD coverage
from `7777770000000000` to `7777775999999999`

---

### OA-3. Catch Me If You Can -- Fraud Detection (3 parts)

**Reported on:** programhelp.net, linkjob.ai, LeetCode

**Part 1 -- Count-based detection:**
- Flag merchants as fraudulent when fraudulent transaction count exceeds MCC-specific thresholds
- Only evaluate after observing minimum transaction volume

**Part 2 -- Percentage-based detection:**
- Replace count thresholds with fraud percentage ratios
- Mark merchants fraudulent when fraud % >= threshold
- Once flagged, status is permanent even if % decreases

**Part 3 -- Dispute resolution:**
- Support DISPUTE commands that overturn specific fraudulent transactions
- Disputed charges don't count toward fraud calculations

**Output:** Lexicographically sorted list of fraudulent merchant account IDs

**Patterns:**
- State machine tracking per merchant
- Streaming/incremental computation
- Three dimensions of state: merchant fraud scores, customer+merchant cumulative counts,
  customer+merchant+hour frequency

---

### OA-4. Store Closing Time Penalty (3 parts)

(Same as Programming Exercise #8 above -- see that section for details)

---

### OA-5. Subscription Notification Scheduler (3 parts)

**Reported on:** programhelp.net, linkjob.ai

(Closely related to Programming Exercise #3, Email Subscription)

**Part 1 -- Basic scheduling:**
- Schedule emails using relative day offsets ("start", negative integers for days before end, "end")
- Output: `"<day>: [<Email Type>] Subscription for <name> (<plan>)"`

**Part 2 -- Plan changes:**
- Handle plan changes with `[Changed]` messages
- Recalculate remaining duration, reschedule future notifications

**Part 3 -- Renewals:**
- Process renewal events extending subscription duration
- Print `[Renewed]` messages, reschedule expiry notifications

---

### OA-6. Payment Card Validation System (4 parts)

**Reported on:** linkjob.ai, programhelp.net

**Part 1 -- Basic VISA validation:**
- Validate 16-digit VISA cards using Luhn algorithm
- Output "VISA" or "INVALID_CHECKSUM"

**Part 2 -- Multi-network validation:**
- Accept 15-16 digit cards, detect network:
  - VISA: 16 digits starting with 4
  - MASTERCARD: 16 digits starting with 51-55
  - AMEX: 15 digits starting with 34 or 37
- Output network name, "INVALID_CHECKSUM", or "UNKNOWN_NETWORK"

**Part 3 -- Redacted cards:**
- Handle cards with `*` representing 1-5 redacted digits
- Count valid cards per network
- Output sorted alphabetically by network

**Part 4 -- Corrupted cards:**
- Handle cards ending with `?` indicating one error:
  single digit changed OR two adjacent digits swapped
- Output all valid original card numbers in numeric order as `"card_number,NETWORK"`

---

### OA-7. Accept-Language Parser

(Same as Programming Exercise #5 above -- see that section for details)

---

## Integration Exercise Problems

These are done during the onsite. You clone a provided GitHub repo and have
full internet access. 1 hour. Focus is on reading docs, calling APIs, and writing
production-quality code.

---

### Integration-1. BikeMap API Integration

**Type:** Integration Exercise (Onsite)
**Reported on:** linkjob.ai, programhelp.net, prachub.com

**Problem:**
- Clone a provided repository
- Read three JSON files and convert to dictionaries
- Perform ETL (Extract-Transform-Load) operations
- Call a specified BikeMap routing API (POST requests)
- Store the returned response data
- Implement several small components

**Structure:** ~5 requirements, candidates typically complete 3-4

**Key skills tested:**
- Git operations
- Reading and following README instructions
- Making HTTP requests (POST)
- JSON serialization/deserialization
- Clean code (descriptive variable names, helper functions)

---

### Integration-2. Stripe Payment API Integration

**Type:** Integration Exercise (Onsite)
**Reported on:** linkjob.ai, blog.rampatra.com

**Problem:**
- Read a file containing request data arrays
- Execute HTTP POST calls with that data to the Stripe API
- Output responses (expected: "200 OK for all requests")
- Provided utilities: file-reading methods, HTTP/JSON parsing libraries

**Key skills tested:**
- API integration
- Error handling
- PCI compliance awareness
- Working with existing utility code

---

### Integration-3. Repository Feature Implementation

**Type:** Integration Exercise (Onsite)
**Reported on:** linkjob.ai, Glassdoor

**Problem:**
- Clone a Git repo
- Call a specified API
- Store results in a database

**Key engineering practices evaluated:**
- Environment variable management (.env files)
- Exception handling for API and database operations
- Transaction usage for database writes
- Comprehensive logging
- Reusing existing API client wrappers

---

## Bug Squash / Debug Round Problems

Candidates clone a fork of a popular open-source project and fix a failing unit test.
45-60 minutes. Two engineers observe. Emphasis on debugging methodology.

---

### Debug-1. Mako Template Engine (Python)

**Reported on:** programhelp.net, linkjob.ai, LeetCode, 1Point3Acres

**Bug 1:** Path handling error -- the code reads a directory path as a file instead
of properly distinguishing between directories and files. Fix: add existence and
file-type checks with clear error messages.

**Bug 2:** Missing AST visitor function -- the code is missing a function that handles
a specific Abstract Syntax Tree (AST) node type, causing runtime errors when Mako
parses certain templates. Fix: add the missing node handler and default exception
handling for unknown nodes.

---

### Debug-2. Moshi JSON Library

**Reported on:** 1Point3Acres

**Problem:** Debug issues in the Moshi JSON parsing library. Involves understanding
JSON serialization/deserialization pipelines and fixing failing test cases.

Details are less publicly documented than the Mako debug round.

---

### Debug-3. SnakeYAML

**Reported on:** linkjob.ai

**Bugs:** Boolean parsing issues and CSV parsing bugs in the YAML parser.

---

### Debug-4. Requests Library (Python)

**Reported on:** blog.rampatra.com, Glassdoor

**Problem:** Check out a fork of the Python `requests` library and fix a broken test.
Navigate the codebase to identify where the bug is introduced and provide a fix.

---

### Debug-5. Generic Open Source Project

**Reported on:** Glassdoor, blog.jez.io

**Format:** "Here's a repo you've never seen before. Here's how to build and run the
tests. There's a bug: what we're observing is X, but we want to see Y instead."

Projects used include well-known open-source libraries in the candidate's chosen language.
The bugs are real GitHub issues that were previously fixed.

---

## stripe-interview GitHub Organization

**URL:** https://github.com/stripe-interview

The organization hosts 10 repositories -- all are environment setup repos to ensure
candidates have their laptops configured before interview day. They contain NO actual
interview problems.

### Repositories:

| Repository | Language | Stars | Description |
|---|---|---|---|
| python-interview-prep | Python | 112 | Python 3.6+, venv setup, TLS verification |
| java-interview-prep | Java | 46 | Maven build (`mvn clean -e install`) |
| javascript-interview-prep | JavaScript | 7 | Node.js 18+, includes `node-fetch` package |
| csharp-interview-prep | C# | 11 | .NET setup |
| scala-interview-prep | Shell/Scala | 9 | SBT project |
| cpp-interview-prep | C++ | 9 | cmake + libcurl required |
| ruby-interview-prep | Ruby | 5 | Bundle + rake spec |
| ml-python-interview-prep | Python | -- | Python 3.6+, ML-specific packages |
| android-kotlin-interview-prep | Kotlin | -- | Android/Kotlin setup |
| react-native-interview-prep | TypeScript | -- | React Native setup |

### What they contain:

Each repo has:
- **README.md** -- Setup instructions (clone, create venv/install deps, verify)
- **Requirements file** -- Language-specific dependencies
- **Verification script** -- Validates TLS 1.2+ support and dependency installation

The Python repo includes: `interview_requirements.txt` and `verify_tls.py`
The JavaScript repo includes: `hello-world.mjs` (fetches data from an open API), `package.json` (depends on `node-fetch`), `.vscode/launch.json` for debugging

**Key takeaway:** These repos signal that the interview involves:
- Making HTTPS requests (TLS verification)
- Working with external APIs (node-fetch, requests)
- Using an IDE with debugging capabilities
- Having a working local development environment

---

## Problem Index by Pattern

### By Data Structure / Algorithm

| Pattern | Problems Using It |
|---|---|
| **Hash Map** | All problems (balance tracking, configuration lookup, rate limiting, etc.) |
| **Graph (adjacency list)** | Currency Conversion, User Deduplication |
| **BFS** | Currency Conversion (Part 3), connected components |
| **DFS** | Currency Conversion (Part 2), Account Balancing optimization |
| **Union-Find** | User Deduplication (Part 3) |
| **LRU Cache** | Account Scheduler (Part 3) |
| **Sliding Window** | Rate Limiter, Fraud Detection |
| **Priority Queue** | Email Subscription Scheduler |
| **Stack** | Brace Expansion, Store Closing (BEGIN/END parsing) |
| **Prefix Sum** | Store Closing Penalty |
| **String Parsing** | Accept-Language, Invoice Reconciliation, Dispute Parser, Atlas Name Check |
| **Luhn Algorithm** | Card Validation, Credit Card Masking |
| **Cartesian Product** | Brace Expansion |
| **Greedy** | Store Closing Penalty, Account Balancing |
| **Sorting** | All output-formatting problems |
| **State Machine** | Fraud Detection, Subscription Scheduler |

### By Topic / Domain

| Domain | Problems |
|---|---|
| **Payments/Transactions** | Transaction Balance, Invoice Reconciliation, Account Balancing, Dispute Parser |
| **Pricing/Billing** | Shipping Calculator, Tiered Pricing |
| **Fraud/Security** | Fraud Detection, Card Validation, Credit Card Masking |
| **Notifications** | Email Subscription Scheduler |
| **Infrastructure** | Rate Limiter, Account Scheduler, Accept-Language Parser |
| **Data Quality** | User Deduplication, Atlas Company Name Check |
| **File/Data Processing** | CSV Processing, Card Range Obfuscation |

### By Round Type

| Round | Problems |
|---|---|
| **Phone Screen** | Shipping Calculator, Currency Conversion, Accept-Language, Store Closing, Min/Max Finder, User Deduplication, Credit Card Masking, Brace Expansion |
| **OA (HackerRank)** | Atlas Name Check, Card Range Obfuscation, Fraud Detection, Store Closing, Subscription Scheduler, Card Validation, Transaction Balance |
| **Onsite Coding** | Email Subscription, Account Scheduler, Account Balancing, Invoice Reconciliation, Transaction Balance |
| **Integration** | BikeMap, Stripe Payment API, Repository Feature Implementation |
| **Bug Squash** | Mako Template, Moshi JSON, SnakeYAML, Requests library |

---

## Sources

- [Stripe Interview Process & Questions -- interviewing.io](https://interviewing.io/stripe-interview-questions)
- [linkjob.ai -- Stripe Technical Interview](https://www.linkjob.ai/interview-questions/stripe-technical-interview/)
- [linkjob.ai -- Stripe Coding Interview 2026](https://www.linkjob.ai/interview-questions/stripe-coding-interview)
- [linkjob.ai -- Stripe HackerRank OA](https://www.linkjob.ai/interview-questions/stripe-hackerrank-online-assessment/)
- [linkjob.ai -- Stripe Interview Questions (8 rounds)](https://www.linkjob.ai/interview-questions/stripe-interview-questions/)
- [linkjob.ai -- Stripe Integration Round](https://www.linkjob.ai/interview-questions/stripe-integration-round/)
- [linkjob.ai -- Stripe Intern Interview](https://www.linkjob.ai/interview-questions/stripe-intern-interview/)
- [linkjob.ai -- Stripe SWE Interview Journey](https://www.linkjob.ai/interview-questions/stripe-software-engineer-interview)
- [programhelp.net -- Stripe OA 6 Classic Problems](https://programhelp.net/en/oa/stripe-oa-interview-questions-experience/)
- [programhelp.net -- Stripe 2026 New Grad OA](https://programhelp.net/en/oa/stripe-2026-new-grad-oa-overview/)
- [programhelp.net -- Stripe HackerRank OA Guide](https://programhelp.net/en/oa/stripe-hackerrank-online-assessment-questions-guide/)
- [programhelp.net -- Stripe VO 5 Rounds](https://programhelp.net/en/vo/stripe-vo-interview-5rounds-experience-guide/)
- [programhelp.net -- Stripe OA Questions](https://programhelp.net/en/stripe-oa-question-stripe-sde/)
- [programhelp.net -- Stripe Interview Question](https://programhelp.net/en/stripe-interview-question-stripe-oa/)
- [codinginterview.com -- Stripe Interview Questions](https://www.codinginterview.com/guide/stripe-interview-questions/)
- [bigtechexperts.com -- Stripe Currency Conversion](https://www.bigtechexperts.com/companies/stripe/swe-algorithm-questions/interview-question-1)
- [staffengprep.com -- Stripe Prep](https://staffengprep.com/companies/stripe/)
- [prepfully.com -- Stripe SWE Guide](https://prepfully.com/interview-guides/stripe-software-engineer)
- [Exponent -- Stripe SWE Guide](https://www.tryexponent.com/guides/stripe-swe-interview)
- [blog.rampatra.com -- Stripe Interview Dublin](https://blog.rampatra.com/stripe-interview-for-software-engineer)
- [Glassdoor -- Stripe SWE Questions](https://www.glassdoor.com/Interview/Stripe-Software-Engineer-Interview-Questions-EI_IE671932.0,6_KO7,24.htm)
- [1Point3Acres -- Stripe Problems (133)](https://www.1point3acres.com/interview/problems/company/stripe)
- [1Point3Acres -- Shipping Cost Phone Screen](https://www.1point3acres.com/interview/thread/1131552)
- [interviewexperiences.in -- Stripe Round 1](https://interviewexperiences.in/experience/stripe/stripe-interview-round-1)
- [Rate Limiters Gist (Paul Tarjan)](https://gist.github.com/ptarjan/e38f45f2dfe601419ca3af937fff574d)
- [Lodely -- Stripe OA 2025](https://www.lodely.com/blog/stripe-online-assessment-2025)
- [interviewquery.com -- Stripe SWE Guide](https://www.interviewquery.com/interview-guides/stripe-software-engineer)
- [interviewquery.com -- Stripe Questions 2026](https://www.interviewquery.com/interview-guides/stripe)
- [GitHub -- stripe-interview org](https://github.com/stripe-interview)
- [LeetCode Discuss -- Stripe Phone Screen](https://leetcode.com/discuss/post/5883672/stripe-phone-screen-by-anonymous_user-0kk5/)
- [Team Blind -- Stripe Interviews](https://www.teamblind.com/post/Stripe-interviews-pgRBVtmo)
- [Team Blind -- Stripe Onsite Experience](https://www.teamblind.com/post/stripe-onsite-interview-experience-n4mqgn4g)
- [algodaily.com -- Stripe](https://algodaily.com/companies/stripe)
- [leetcodewizard.io -- Stripe SWE Guide](https://leetcodewizard.io/blog/mastering-the-stripe-software-engineer-interview-questions-process-and-expert-tips-for-preparation)
- [Jointaro -- Stripe Non-LeetCode Prep](https://www.jointaro.com/question/m0jSJXuuRTwHtuWguUZc/how-to-prepare-for-non-leetcode-coding-interview-like-stripe-and-reddit/)
- [DEV.to -- Stripe SDE 5-Round VO](https://dev.to/net_programhelp_e160eef28/stripe-sde-five-round-vo-interview-experience-real-questions-tips-4ibo)
