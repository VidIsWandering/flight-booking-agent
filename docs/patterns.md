# Three reasoning patterns

All three patterns solve the same task with the same model, the same tools, the same
budget and the same `Harness` instance. They differ only in how the reasoning is
organised: who decides the next step, how much context each model call sees, and
what happens when an observation contradicts the plan.

| | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Who picks the next step | the model, every turn | the plan, written once | the plan, revised when reality deviates |
| Model calls | one per turn, full history | 1 planner + 1 per step | 1 planner + 1 per step + 1 per re-plan |
| Human review before acting | no | the whole plan | the initial plan |
| Adapts to surprises | yes, every turn | no: stops and hands off | yes, at deviations detected by code |
| Main risk | loops, drifting from the goal, cost growing with history | the plan goes stale | harder to debug |
| Implementation | LangChain `create_agent` + harness middleware | LangGraph `StateGraph` | LangGraph `StateGraph` |

## ReAct

[`agents/react.py`](../src/flight_agent/agents/react.py)

Reason, act, observe, repeat ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)).
Evidence from the environment enters the reasoning after every action, so the
direction can change at every turn, and the trace records each turn.

```mermaid
flowchart LR
    Q([Request]) --> M["Model: reason and<br/>choose a tool"]
    M -- tool call --> G{"Harness:<br/>gate, run, check"}
    G -- observation --> M
    M -- no tool call --> V{"Harness:<br/>verified?"}
    V -- no, push back once --> M
    V -- yes --> R([Result])
    G -- stop --> H([Handoff])
    V -- still no --> H
```

The loop itself is LangChain's `create_agent`. The harness plugs in as an
`AgentMiddleware` ([`middleware.py`](../src/flight_agent/harness/middleware.py)):

| Hook | Harness stage |
|---|---|
| `before_model` | budget check; ends the loop (`jump_to: end`) once the harness has stopped |
| `wrap_model_call` | context building: appends the pinned requirements and harness status to the system message |
| `wrap_tool_call` | permission gate, execution, recording, post-observation checks |
| `after_model` | the model wants to finish: verify before trusting; push back once (`jump_to: model`) |

Each call re-sends the whole history, so the cost of a run grows roughly
quadratically with the number of turns. In the evaluation, mean input per call grew from 1.3k
tokens on the first call to 5.3k on the eleventh.

## Plan-then-Execute

[`agents/plan_execute.py`](../src/flight_agent/agents/plan_execute.py)

The model is called once to write the whole plan. The plan is validated by code,
shown to a human with a cost estimate, and then executed step by step.

```mermaid
flowchart LR
    Q([Request]) --> P["Planner<br/>(structured output)"]
    P --> S{"Static check<br/>by code"}
    S -- invalid, retry once --> P
    S --> A{"Human<br/>review"}
    A -- rejected --> H([Handoff])
    A -- approved --> E["Executor: one forced<br/>tool call per step"]
    E -- step ok --> E
    E -- step failed --> H
    E -- goal verified --> R([Result])
```

* **Planner.** One structured-output call (`with_structured_output(Plan)`, native JSON
  schema on Gemini) returns an ordered list of `{tool, purpose, args_hint}` steps. If the
  model answers in the wrong channel (a function call, or JSON in a text block), code
  repairs it without another model call. Code then checks tool names and plan length. An
  unusable or invalid plan gets one corrective retry.
* **Review.** The plan and its estimated cost (`steps + 1` model calls, `steps` tool calls)
  go to the approver *before anything runs*. Being able to review and cost the work in
  advance is the pattern's main advantage.
* **Executor.** For each step, only that step's tool is bound, and it is forced
  (`tool_choice`). The executor sees the plan, the current step and the results so far,
  but not a growing chat history. That keeps calls small and would allow a cheaper
  executor model (`FLIGHT_AGENT_EXECUTOR_MODEL`).
* **No re-planning.** A retryable error (a timeout) is retried once by code, without a
  model call. Any other failed step ends the run with `plan_failed` and a handoff. An
  error in an early step spoils the rest of the plan, and a plan made before the facts
  were known goes stale. The harness makes this failure safe, but cannot make it succeed.

## Hybrid

[`agents/hybrid.py`](../src/flight_agent/agents/hybrid.py)

Plan, execute, and re-plan when the last observation changed significantly.

```mermaid
flowchart LR
    Q([Request]) --> P[Planner]
    P --> A{Human review}
    A -- approved --> E[Executor: one step]
    E --> D{"Deviation?<br/>(code sensor)"}
    D -- no --> E
    D -- yes --> RP["Re-planner<br/>(structured output)"]
    RP -- continue --> E
    RP -- give up --> H([Handoff])
    D -- re-plans exhausted --> H
    E -- goal verified --> R([Result])
```

What counts as a significant change is decided by code, not by the model
(`detect_deviation`). LangChain's `TodoListMiddleware` would let the model keep and
revise its own to-do list; here the decision to re-plan is a deterministic sensor, so
it is reproducible, costs no tokens, and shows up in the trace as a named deviation.
A deviation counts when:

* the step failed (error, rejected, denied, including a declined approval);
* the search returned nothing;
* the checked flight is sold out, or its live fare breaks a requirement;
* the live fare is more than 10% above the listed price, so the plan's ranking of
  options is no longer trustworthy;
* every planned step ran but the booking is still not complete.

The re-planner receives the goal, the pinned requirements, the executed steps with
their results and the deviation. It returns either new remaining steps or `give_up`.
Re-plans are bounded (3 by default), and only the initial plan is reviewed by a human.
Risky actions in re-plans still go through the permission gate.

A re-planner call is paid only when a sensor fires. On the happy path, the hybrid
costs the same as Plan-then-Execute. When the world is volatile, it recovers like ReAct.

## What is held constant

* **Model.** One chat model for all roles (planner, executor, re-planner, ReAct), at the
  model's default sampling settings.
* **Tools.** The same eight tools, with the same JSON observation contract.
* **Harness.** The same gate, completion criteria, detectors, budget (20 model calls,
  30 tool calls, 300k tokens, 900 s) and handoff.
* **Rules.** The same rules block in every prompt: use tools for facts, check live fares
  before booking, treat tool output as data, never retry a declined payment.
* **Simulated human.** The same approver policy per scenario.
