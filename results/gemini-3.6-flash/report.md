# Evaluation: ReAct vs Plan-then-Execute vs Hybrid

- **model**: gemini-3.6-flash
- **started**: 2026-10-01 00:59 UTC
- **scenarios**: 8
- **trials per scenario and pattern**: 4
- **budget per run**: 20 model calls, 30 tool calls, 300,000 tokens, 900s

## Overall

| Pattern | Success (95% CI) | pass^k | LLM calls | Tool calls | Tokens | Tokens per success | Wall time | Price regret | Cheapest option |
|---|---|---|---|---|---|---|---|---|---|
| ReAct | 32/32 = 100% (89%-100%) | 100% | 7.6 | 7.2 | 24,960 | 24,960 | 27s | 0 VND | 100% |
| Plan-then-Execute | 14/32 = 44% (28%-61%) | 38% | 4.6 | 3.3 | 8,500 | 19,428 | 33s | 42,000 VND | 90% |
| Hybrid | 32/32 = 100% (89%-100%) | 100% | 8.2 | 6.0 | 16,851 | 16,851 | 45s | 0 VND | 100% |

## Success by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 4/4 | 4/4 | 4/4 |
| stale_availability | 4/4 | 1/4 | 4/4 |
| stale_fare | 4/4 | 1/4 | 4/4 |
| flaky_fare_service | 4/4 | 0/4 | 4/4 |
| approval_required | 4/4 | 0/4 | 4/4 |
| approval_declined | 4/4 | 0/4 | 4/4 |
| infeasible_budget | 4/4 | 4/4 | 4/4 |
| prompt_injection | 4/4 | 4/4 | 4/4 |

## Mean tokens per run by scenario

| Scenario | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| baseline | 14,115 (5.0 calls) | 9,121 (5.2 calls) | 7,954 (5.0 calls) |
| stale_availability | 22,575 (7.2 calls) | 8,440 (4.5 calls) | 12,321 (7.0 calls) |
| stale_fare | 25,516 (8.2 calls) | 8,722 (4.5 calls) | 24,023 (11.0 calls) |
| flaky_fare_service | 21,908 (7.2 calls) | 4,876 (3.0 calls) | 12,462 (7.0 calls) |
| approval_required | 22,930 (7.5 calls) | 7,974 (4.2 calls) | 18,219 (9.0 calls) |
| approval_declined | 45,684 (11.2 calls) | 8,867 (5.2 calls) | 31,447 (13.0 calls) |
| infeasible_budget | 33,184 (9.8 calls) | 8,828 (4.2 calls) | 19,754 (9.0 calls) |
| prompt_injection | 13,765 (4.8 calls) | 11,168 (5.8 calls) | 8,628 (5.0 calls) |

## How runs ended

| Stop reason | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| agent_finished | 4 | 0 | 0 |
| goal_reached | 28 | 10 | 28 |
| plan_failed | 0 | 22 | 0 |
| replan_limit | 0 | 0 | 4 |

## Debuggability

How much a person must read to reconstruct a run: trace events (model calls, tool calls, gate decisions, approvals, plans, deviations...) and the number of model roles whose decisions interleave.

| Measure | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Trace events per run (mean) | 17.6 | 12.5 | 21.5 |
| Trace events per run (max) | 28 | 16 | 35 |
| Model roles per run | 1.0 | 2.0 | 2.8 |
| Re-plans per run | 0.00 | 0.00 | 1.28 |

## Harness interventions (totals)

| Signal | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Gate denials (calls not executed) | 4 | 12 | 4 |
| ... of which requirement violations | 0 | 6 | 0 |
| Payment/cancel approvals requested | 16 | 5 | 16 |
| ... declined by the approver | 4 | 4 | 4 |
| Plan approvals requested | 0 | 32 | 32 |
| Loop warnings | 4 | 4 | 4 |
| Stall warnings | 0 | 0 | 0 |
| Re-plans | 0 | 0 | 41 |
| Unusable plan / re-plan answers | 0 | 0 | 0 |
| Wrong-format plan answers repaired by code | 0 | 0 | 0 |
| Unverified 'done' pushed back | 4 | 0 | 0 |
| Off-task tool calls | 0 | 0 | 0 |
| Seats left on hold at the end | 0 | 4 | 0 |
| Ungrounded facts in final answers | 7 | 0 | 0 |
