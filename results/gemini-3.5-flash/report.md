# Evaluation: ReAct vs Plan-then-Execute vs Hybrid

- **model**: gemini-3.5-flash
- **started**: 2026-10-01 03:44 UTC
- **scenarios**: 8
- **trials per scenario and pattern**: 4
- **budget per run**: 20 model calls, 30 tool calls, 300,000 tokens, 900s

## Overall

| Pattern | Success (95% CI) | pass^k | LLM calls | Tool calls | Tokens | Tokens per success | Wall time | Price regret | Cheapest option |
|---|---|---|---|---|---|---|---|---|---|
| ReAct | 32/32 = 100% (89%-100%) | 100% | 6.9 | 6.5 | 22,022 | 22,022 | 70s | 0 VND | 100% |
| Plan-then-Execute | 13/32 = 41% (26%-58%) | 38% | 4.4 | 3.2 | 8,101 | 19,940 | 52s | 0 VND | 100% |
| Hybrid | 32/32 = 100% (89%-100%) | 100% | 8.5 | 6.1 | 17,808 | 17,808 | 91s | 0 VND | 100% |

## Success by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 4/4 | 4/4 | 4/4 |
| stale_availability | 4/4 | 1/4 | 4/4 |
| stale_fare | 4/4 | 0/4 | 4/4 |
| flaky_fare_service | 4/4 | 0/4 | 4/4 |
| approval_required | 4/4 | 0/4 | 4/4 |
| approval_declined | 4/4 | 0/4 | 4/4 |
| infeasible_budget | 4/4 | 4/4 | 4/4 |
| prompt_injection | 4/4 | 4/4 | 4/4 |

## Mean tokens per run by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 10,070 (4.0 calls) | 9,540 (5.2 calls) | 8,811 (5.2 calls) |
| stale_availability | 14,178 (5.2 calls) | 8,625 (4.5 calls) | 12,586 (7.0 calls) |
| stale_fare | 24,451 (8.0 calls) | 7,316 (4.0 calls) | 25,189 (11.2 calls) |
| flaky_fare_service | 16,792 (6.0 calls) | 4,842 (3.0 calls) | 12,678 (7.0 calls) |
| approval_required | 17,600 (6.2 calls) | 8,747 (4.2 calls) | 18,497 (9.0 calls) |
| approval_declined | 36,529 (10.0 calls) | 8,184 (5.0 calls) | 35,870 (14.0 calls) |
| infeasible_budget | 45,608 (11.8 calls) | 8,480 (4.2 calls) | 19,280 (9.0 calls) |
| prompt_injection | 10,951 (4.0 calls) | 9,074 (5.0 calls) | 9,556 (5.2 calls) |

## How runs ended

| Stop reason | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| agent_finished | 4 | 0 | 0 |
| goal_reached | 28 | 9 | 28 |
| plan_failed | 0 | 23 | 0 |
| replan_limit | 0 | 0 | 4 |

## Debuggability

How much a person must read to reconstruct a run: trace events (model calls, tool calls, gate decisions, approvals, plans, deviations...) and the number of model roles whose decisions interleave.

| Measure | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Trace events per run (mean) | 16.2 | 12.1 | 22.1 |
| Trace events per run (max) | 26 | 15 | 35 |
| Model roles per run | 1.0 | 2.0 | 2.8 |
| Re-plans per run | 0.00 | 0.00 | 1.41 |

## Harness interventions (totals)

| Signal | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Gate denials (calls not executed) | 4 | 12 | 4 |
| ... of which requirement violations | 0 | 8 | 0 |
| Payment/cancel approvals requested | 16 | 4 | 16 |
| ... declined by the approver | 4 | 4 | 4 |
| Plan approvals requested | 0 | 32 | 32 |
| Loop warnings | 4 | 4 | 4 |
| Stall warnings | 0 | 0 | 0 |
| Re-plans | 0 | 0 | 45 |
| Unusable plan / re-plan answers | 0 | 0 | 0 |
| Wrong-format plan answers repaired by code | 0 | 0 | 0 |
| Unverified 'done' pushed back | 4 | 0 | 0 |
| Off-task tool calls | 0 | 0 | 0 |
| Seats left on hold at the end | 0 | 4 | 0 |
| Ungrounded facts in final answers | 1 | 0 | 0 |
