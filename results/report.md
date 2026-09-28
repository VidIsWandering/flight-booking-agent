# Evaluation: ReAct vs Plan-then-Execute vs Hybrid

- **serving models**: gemini-3.5-flash, gemini-3.6-flash
- **trials per scenario and pattern**: 8 (each trial served by one model)
- **runs**: 192

## Consistency across serving models

Successful runs per pattern, for each model that served the agent.

| Model | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| gemini-3.5-flash | 32/32 | 13/32 | 32/32 |
| gemini-3.6-flash | 32/32 | 14/32 | 32/32 |

## Overall

| Pattern | Success (95% CI) | pass^k | LLM calls | Tool calls | Tokens | Tokens per success | Wall time | Price regret | Cheapest option |
|---|---|---|---|---|---|---|---|---|---|
| ReAct | 64/64 = 100% (94%-100%) | 100% | 7.3 | 6.9 | 23,491 | 23,491 | 49s | 0 VND | 100% |
| Plan-then-Execute | 27/64 = 42% (31%-54%) | 38% | 4.5 | 3.2 | 8,300 | 19,675 | 43s | 22,105 VND | 95% |
| Hybrid | 64/64 = 100% (94%-100%) | 100% | 8.4 | 6.0 | 17,330 | 17,330 | 68s | 0 VND | 100% |

## Success by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 8/8 | 8/8 | 8/8 |
| stale_availability | 8/8 | 2/8 | 8/8 |
| stale_fare | 8/8 | 1/8 | 8/8 |
| flaky_fare_service | 8/8 | 0/8 | 8/8 |
| approval_required | 8/8 | 0/8 | 8/8 |
| approval_declined | 8/8 | 0/8 | 8/8 |
| infeasible_budget | 8/8 | 8/8 | 8/8 |
| prompt_injection | 8/8 | 8/8 | 8/8 |

## Mean tokens per run by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 12,093 (4.5 calls) | 9,331 (5.2 calls) | 8,382 (5.1 calls) |
| stale_availability | 18,376 (6.2 calls) | 8,533 (4.5 calls) | 12,453 (7.0 calls) |
| stale_fare | 24,983 (8.1 calls) | 8,019 (4.2 calls) | 24,606 (11.1 calls) |
| flaky_fare_service | 19,350 (6.6 calls) | 4,859 (3.0 calls) | 12,570 (7.0 calls) |
| approval_required | 20,265 (6.9 calls) | 8,360 (4.2 calls) | 18,358 (9.0 calls) |
| approval_declined | 41,106 (10.6 calls) | 8,525 (5.1 calls) | 33,659 (13.5 calls) |
| infeasible_budget | 39,396 (10.8 calls) | 8,654 (4.2 calls) | 19,517 (9.0 calls) |
| prompt_injection | 12,358 (4.4 calls) | 10,121 (5.4 calls) | 9,092 (5.1 calls) |

## How runs ended

| Stop reason | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| agent_finished | 8 | 0 | 0 |
| goal_reached | 56 | 19 | 56 |
| plan_failed | 0 | 45 | 0 |
| replan_limit | 0 | 0 | 8 |

## Debuggability

How much a person must read to reconstruct a run: trace events (model calls, tool calls, gate decisions, approvals, plans, deviations...) and the number of model roles whose decisions interleave.

| Measure | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Trace events per run (mean) | 16.9 | 12.3 | 21.8 |
| Trace events per run (max) | 28 | 16 | 35 |
| Model roles per run | 1.0 | 2.0 | 2.8 |
| Re-plans per run | 0.00 | 0.00 | 1.34 |

## Harness interventions (totals)

| Signal | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Gate denials (calls not executed) | 8 | 24 | 8 |
| ... of which requirement violations | 0 | 14 | 0 |
| Payment/cancel approvals requested | 32 | 9 | 32 |
| ... declined by the approver | 8 | 8 | 8 |
| Plan approvals requested | 0 | 64 | 64 |
| Loop warnings | 8 | 8 | 8 |
| Stall warnings | 0 | 0 | 0 |
| Re-plans | 0 | 0 | 86 |
| Unusable plan / re-plan answers | 0 | 0 | 0 |
| Wrong-format plan answers repaired by code | 0 | 0 | 0 |
| Unverified 'done' pushed back | 8 | 0 | 0 |
| Off-task tool calls | 0 | 0 | 0 |
| Seats left on hold at the end | 0 | 8 | 0 |
| Ungrounded facts in final answers | 8 | 0 | 0 |
