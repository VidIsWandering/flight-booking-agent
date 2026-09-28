# Recorded evaluation runs

The evaluation analysed in [docs/evaluation.md](../docs/evaluation.md): 8 scenarios ×
3 patterns × 8 trials, each trial served entirely by one Gemini flash model.

| Path | Content |
|---|---|
| `report.md`, `summary.json` | The pooled report over all serving models: consistency per model, overall comparison, success by scenario, tokens by scenario, stop reasons, debuggability, harness interventions |
| `<model>/meta.json` | Model, start time, number of trials, per-run budget |
| `<model>/runs.jsonl` | One JSON line per run: outcome, booked flight, price regret, stop reason, handoff, metrics |
| `<model>/traces/<scenario>__<pattern>__t<trial>.jsonl` | One JSON line per model call, tool call and harness decision |
| `<model>/report.md`, `<model>/summary.json` | The same tables for the runs of one model |

Rebuild every report, or read one run back, without calling any model:

```bash
uv run flight-agent report results/gemini-* --pooled results
uv run flight-agent replay -s approval_declined -p hybrid --trial 1 --model gemini-3.6-flash
```
