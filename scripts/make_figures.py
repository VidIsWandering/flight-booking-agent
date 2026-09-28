"""Draw the evaluation figures in docs/images/ from the recorded results.

    uv run --group figures python scripts/make_figures.py

Every number comes from results/ (the pooled summary and the traces), so the figures
always match the recorded runs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "images"

PATTERNS = ["react", "plan_execute", "hybrid"]
LABELS = {"react": "ReAct", "plan_execute": "Plan-then-Execute", "hybrid": "Hybrid"}
COLORS = {"react": "#2a78d6", "plan_execute": "#eb6834", "hybrid": "#1baf7a"}
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ecebe6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "legend.frameon": False,
        "savefig.dpi": 200,
    }
)


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def scenario_order(summary: dict) -> list[str]:
    return list(summary["scenarios"])


def success_by_scenario(summary: dict) -> None:
    scenarios = scenario_order(summary)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    height = 0.26
    for i, p in enumerate(PATTERNS):
        ys = [k + (1 - i) * (height + 0.02) for k in range(len(scenarios))]
        xs = [summary["scenarios"][s][p]["successes"] for s in scenarios]
        ax.barh(ys, xs, height, color=COLORS[p], label=LABELS[p])
        for y, x in zip(ys, xs, strict=True):
            ax.text(x + 0.08, y, f"{x:.0f}", va="center", fontsize=7, color=INK2)
    runs = summary["scenarios"][scenarios[0]][PATTERNS[0]]["runs"]
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_xlim(0, runs + 0.6)
    ax.set_xticks(range(0, int(runs) + 1, 2))
    ax.set_xlabel(f"successful runs out of {runs:.0f}")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "success_by_scenario.png")
    plt.close(fig)


def tokens_run_vs_success(summary: dict) -> None:
    by_pattern = {s["pattern"]: s for s in summary["patterns"]}
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    width = 0.34
    for i, p in enumerate(PATTERNS):
        s = by_pattern[p]
        for j, (value, hatch) in enumerate(
            [(s["mean_total_tokens"], None), (s["tokens_per_success"], "////")]
        ):
            x = i + (j - 0.5) * (width + 0.04)
            ax.bar(
                x,
                value,
                width,
                color=COLORS[p] if hatch is None else "#ffffff",
                edgecolor=COLORS[p],
                hatch=hatch,
                linewidth=1.2,
            )
            ax.text(
                x,
                value + 400,
                f"{value / 1000:.1f}k",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(range(3), [LABELS[p] for p in PATTERNS])
    ax.set_ylabel("tokens")
    ax.set_ylim(0, 27000)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.legend(
        handles=[
            Patch(facecolor="#9a9994", edgecolor="#9a9994", label="Mean tokens per run"),
            Patch(
                facecolor="#ffffff",
                edgecolor="#9a9994",
                hatch="////",
                label="Tokens per successful run",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUT / "tokens_per_run_and_success.png")
    plt.close(fig)


def tokens_by_scenario(summary: dict) -> None:
    scenarios = scenario_order(summary)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.26
    for i, p in enumerate(PATTERNS):
        xs = [k + (i - 1) * (width + 0.02) for k in range(len(scenarios))]
        ys = [summary["scenarios"][s][p]["mean_tokens"] for s in scenarios]
        ax.bar(xs, ys, width, color=COLORS[p], label=LABELS[p], edgecolor="#ffffff", linewidth=0.6)
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("mean tokens per run")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.legend(loc="upper left", fontsize=8, ncols=3)
    ax.set_ylim(0, 47000)
    fig.tight_layout()
    fig.savefig(OUT / "tokens_by_scenario.png")
    plt.close(fig)


def context_growth() -> None:
    inputs: dict[tuple[str, int], list[int]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*/traces/*.jsonl")):
        index: dict[str, int] = defaultdict(int)
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["kind"] != "model_call":
                continue
            role = event["data"]["role"]
            index[role] += 1
            inputs[(role, index[role])].append(event["data"].get("input_tokens", 0))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    series = [
        ("react", "ReAct (whole history)", COLORS["react"]),
        ("executor", "Executor (plan + results)", COLORS["plan_execute"]),
    ]
    for role, label, color in series:
        points = sorted(
            (i, sum(v) / len(v)) for (r, i), v in inputs.items() if r == role and i <= 11
        )
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5)
        ax.text(
            xs[-1] + 0.25,
            ys[-1],
            f"{label}\n{ys[-1] / 1000:.1f}k",
            color=INK,
            fontsize=8,
            va="center",
        )
    ax.set_xlabel("model call within a run")
    ax.set_ylabel("mean input tokens")
    ax.set_xticks(range(1, 12))
    ax.set_xlim(0.6, 15.2)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.1f}k")
    fig.tight_layout()
    fig.savefig(OUT / "context_growth.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    success_by_scenario(summary)
    tokens_run_vs_success(summary)
    tokens_by_scenario(summary)
    context_growth()
    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
