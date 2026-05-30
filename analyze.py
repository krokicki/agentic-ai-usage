#!/usr/bin/env python3
"""Generate charts summarizing AI coding-agent token usage.

Reads agent logs directly — no external tools required:
  * Claude Code : ~/.claude/projects/**/*.jsonl   (incl. nested subagent logs)
  * Codex       : ~/.codex/sessions/**/rollout-*.jsonl

Every chart plots time on the X axis and tokens on the Y axis.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------
# shared styling
# ----------------------------------------------------------------------------
FAMILIES = ["Opus", "Sonnet", "Haiku", "Codex"]
MODEL_COLORS = {"Opus": "#d97757", "Sonnet": "#6e8fd4",
                "Haiku": "#7ec699", "Codex": "#b083d9"}
PALETTE = ["#d97757", "#6e8fd4", "#7ec699", "#b083d9", "#e0b341", "#5fb9c4",
           "#e0739b", "#8c9eff", "#9ccc65", "#ff8a65", "#4db6ac", "#7e8a99"]

# Per-family list price, USD per million tokens: (input, output).
# Cache write is billed at 1.25x input and cache read at 0.10x input across
# Anthropic models. Codex is set to zero because we don't pay for it directly.
PRICES = {
    "Opus":   (15.0, 75.0),
    "Sonnet": (3.0,  15.0),
    "Haiku":  (1.0,   5.0),
    "Codex":  (0.0,   0.0),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

# List prices above overestimate our actual billing (subscription / discounted
# rates). This factor scales Claude costs to match a known invoice: May 2026
# Claude usage billed $961.90. Recompute if you recalibrate against a new bill.
CLAUDE_CALIBRATION = 0.2222387723400814

# Token-type breakdown used by the cost chart.
COST_COMPONENTS = ["Input", "Output", "Cache write", "Cache read"]
COMPONENT_COLORS = {"Input": "#6e8fd4", "Output": "#d97757",
                    "Cache write": "#e0b341", "Cache read": "#7ec699"}


def cost_components(r):
    """Estimated USD cost of one usage record, split by token type."""
    pin, pout = PRICES.get(r["family"], PRICES["Codex"])
    cal = 1.0 if r["family"] == "Codex" else CLAUDE_CALIBRATION
    pin, pout = pin * cal, pout * cal
    return {
        "Input": r["input"] * pin / 1e6,
        "Output": r["output"] * pout / 1e6,
        "Cache write": r["cache_create"] * pin * CACHE_WRITE_MULT / 1e6,
        "Cache read": r["cache_read"] * pin * CACHE_READ_MULT / 1e6,
    }


def record_cost(r):
    """Total estimated USD cost of one usage record."""
    return sum(cost_components(r).values())


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
        "ytick.color": "#8b949e", "axes.edgecolor": "#30363d",
        "font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
    })


def fmt_tokens(x, _=None):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def fmt_dollars(x, _=None):
    if x >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:.0f}"


def style_axis(ax):
    ax.grid(axis="y", color="#30363d", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_ylabel("Tokens")


# ----------------------------------------------------------------------------
# log parsing  ->  flat list of usage records {date, project, family, tokens}
# ----------------------------------------------------------------------------
def model_family(name):
    if name.startswith("claude-opus"):
        return "Opus"
    if name.startswith("claude-sonnet"):
        return "Sonnet"
    if name.startswith("claude-haiku"):
        return "Haiku"
    return "Codex"  # gpt-*, and any other non-Claude vendor


def parse_claude(claude_dir):
    """Yield usage records from Claude Code logs (recursive; deduplicated)."""
    seen = set()
    pattern = os.path.join(claude_dir, "**", "*.jsonl")
    for jf in glob.glob(pattern, recursive=True):
        for line in open(jf, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            msg = o.get("message") or {}
            u = msg.get("usage")
            ts = o.get("timestamp")
            if not u or not ts:
                continue
            mid, rid = msg.get("id"), o.get("requestId")
            key = (mid, rid) if (mid and rid) else o.get("uuid")
            if key in seen:
                continue
            seen.add(key)
            inp = u.get("input_tokens", 0)
            out = u.get("output_tokens", 0)
            cc = u.get("cache_creation_input_tokens", 0)
            cr = u.get("cache_read_input_tokens", 0)
            tok = inp + out + cc + cr
            if tok <= 0:
                continue
            cwd = o.get("cwd")
            yield {
                "date": ts[:10],
                "project": os.path.basename(cwd) if cwd else "unknown",
                "family": model_family(msg.get("model", "")),
                "tokens": tok,
                "input": inp, "output": out,
                "cache_create": cc, "cache_read": cr,
            }


def parse_codex(codex_dir):
    """Yield usage records from Codex session rollouts.

    Token usage lives in `token_count` events; `last_token_usage` is the
    per-turn delta, so summing it across a session avoids double-counting the
    cumulative `total_token_usage`.
    """
    pattern = os.path.join(codex_dir, "**", "rollout-*.jsonl")
    for jf in glob.glob(pattern, recursive=True):
        cwd = None
        for line in open(jf, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            payload = o.get("payload") or {}
            if o.get("type") == "session_meta":
                cwd = payload.get("cwd")
                continue
            if payload.get("type") != "token_count":
                continue
            last = (payload.get("info") or {}).get("last_token_usage") or {}
            tok = last.get("total_tokens", 0)
            ts = o.get("timestamp")
            if tok <= 0 or not ts:
                continue
            # `input_tokens` includes the cached portion; split it out so the
            # cached part can be priced at the cheaper cache-read rate.
            cached = last.get("cached_input_tokens", 0)
            yield {
                "date": ts[:10],
                "project": os.path.basename(cwd) if cwd else "unknown",
                "family": "Codex",
                "tokens": tok,
                "input": max(last.get("input_tokens", 0) - cached, 0),
                "output": last.get("output_tokens", 0),
                "cache_create": 0,
                "cache_read": cached,
            }


def collect_usage(claude_dir, codex_dir):
    records = []
    if os.path.isdir(claude_dir):
        records.extend(parse_claude(claude_dir))
    else:
        print(f"  note: {claude_dir} not found — skipping Claude", file=sys.stderr)
    if codex_dir and os.path.isdir(codex_dir):
        records.extend(parse_codex(codex_dir))
    else:
        print(f"  note: {codex_dir} not found — skipping Codex", file=sys.stderr)
    if not records:
        sys.exit("No usage logs found.")
    return records


# ----------------------------------------------------------------------------
# aggregation helpers
# ----------------------------------------------------------------------------
def by_period_family(records, key):
    """period -> {family: tokens}. key(date) selects month ('%Y-%m') or day."""
    agg = defaultdict(lambda: defaultdict(int))
    for r in records:
        agg[key(r["date"])][r["family"]] += r["tokens"]
    return agg


def cost_by_period_component(records, key):
    """period -> {token-type: USD}. key(date) selects month ('%Y-%m') or day."""
    agg = defaultdict(lambda: defaultdict(float))
    for r in records:
        bucket = agg[key(r["date"])]
        for comp, usd in cost_components(r).items():
            bucket[comp] += usd
    return agg


def by_date_project(records):
    agg = defaultdict(lambda: defaultdict(int))
    proj_total = defaultdict(int)
    for r in records:
        agg[r["date"]][r["project"]] += r["tokens"]
        proj_total[r["project"]] += r["tokens"]
    return agg, proj_total


# ----------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------
def chart_monthly(records, out_dir):
    agg = by_period_family(records, lambda d: d[:7])
    months = sorted(agg)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    bottom = np.zeros(len(months))
    used = []
    for f in FAMILIES:
        vals = np.array([agg[m].get(f, 0) for m in months])
        if vals.sum() == 0:
            continue
        ax.bar(months, vals, bottom=bottom, color=MODEL_COLORS[f],
               width=0.62, label=f)
        bottom += vals
        used.append(f)
    for i, t in enumerate(bottom):
        ax.text(i, t, " " + fmt_tokens(t), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#e6edf3")
    style_axis(ax)
    _legend(ax, used, MODEL_COLORS)
    ax.set_title("AI Token Usage — Monthly (stacked by model)", loc="left", pad=14)
    ax.set_ylim(0, bottom.max() * 1.12)
    _save(fig, out_dir, "usage_monthly.png")


def chart_cost(records, out_dir):
    agg = cost_by_period_component(records, lambda d: d[:7])
    months = sorted(agg)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    bottom = np.zeros(len(months))
    used = []
    for comp in COST_COMPONENTS:
        vals = np.array([agg[m].get(comp, 0.0) for m in months])
        if vals.sum() == 0:
            continue
        ax.bar(months, vals, bottom=bottom, color=COMPONENT_COLORS[comp],
               width=0.62, label=comp)
        bottom += vals
        used.append(comp)
    for i, t in enumerate(bottom):
        ax.text(i, t, " " + fmt_dollars(t), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#e6edf3")
    ax.grid(axis="y", color="#30363d", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_dollars))
    ax.set_ylabel("Estimated cost (USD)")
    _legend(ax, used, COMPONENT_COLORS)
    ax.set_title("AI Cost — Monthly est. (stacked by token type)",
                 loc="left", pad=14)
    ax.set_ylim(0, bottom.max() * 1.12)
    _save(fig, out_dir, "usage_cost_monthly.png")


def chart_daily(records, out_dir):
    agg = by_period_family(records, lambda d: d)
    dates = sorted(agg)
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=160)
    bottom = np.zeros(len(dts))
    used = []
    for f in FAMILIES:
        vals = np.array([agg[d].get(f, 0) for d in dates])
        if vals.sum() == 0:
            continue
        ax.bar(dts, vals, bottom=bottom, color=MODEL_COLORS[f], width=0.9, label=f)
        bottom += vals
        used.append(f)
    style_axis(ax)
    _legend(ax, used, MODEL_COLORS)
    _date_axis(ax, dts)
    ax.set_title("AI Token Usage — Daily (stacked by model)", loc="left", pad=14)
    _save(fig, out_dir, "usage_daily.png")


def chart_rolling(records, out_dir, window=3):
    agg = by_period_family(records, lambda d: d)
    dates = sorted(agg)
    start = datetime.strptime(dates[0], "%Y-%m-%d")
    end = datetime.strptime(dates[-1], "%Y-%m-%d")
    ndays = (end - start).days + 1
    alld = [start + timedelta(days=i) for i in range(ndays)]
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(alld)}
    series = {f: np.zeros(ndays) for f in FAMILIES}
    total = np.zeros(ndays)
    for d in dates:
        i = idx[d]
        for f, t in agg[d].items():
            series[f][i] += t
            total[i] += t

    def roll(a):
        return np.convolve(a, np.ones(window) / window, mode="same")

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=160)
    ax.plot(alld, roll(total), color="#58a6ff", lw=2.6, label="Total", zorder=5)
    ax.fill_between(alld, roll(total), color="#58a6ff", alpha=0.08)
    for f in FAMILIES:
        if series[f].sum() == 0:
            continue
        ax.plot(alld, roll(series[f]), color=MODEL_COLORS[f], lw=1.8,
                label=f, alpha=0.95)
    style_axis(ax)
    ax.set_ylabel(f"Tokens / day ({window}-day avg)")
    ax.legend(loc="upper left", frameon=False, labelcolor="#e6edf3",
              fontsize=11, ncol=2)
    _date_axis(ax, alld, minor=False)
    ax.set_title(f"AI Token Usage — {window}-Day Rolling Average",
                 loc="left", pad=14)
    _save(fig, out_dir, "usage_rolling3d.png")


def chart_projects(records, out_dir, top_n=11):
    agg, proj_total = by_date_project(records)
    ranked = sorted(proj_total.items(), key=lambda x: -x[1])
    top = [p for p, _ in ranked[:top_n]]
    order = top + (["other"] if len(ranked) > top_n else [])
    dates = sorted(agg)
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    series = {p: np.zeros(len(dates)) for p in order}
    for i, d in enumerate(dates):
        for proj, tok in agg[d].items():
            series[proj if proj in top else "other"][i] += tok
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(order)}
    if "other" in series:
        colors["other"] = "#5a6473"

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=160)
    other_label = f"other ({len(ranked) - top_n} projects)"
    bottom = np.zeros(len(dts))
    for p in order:
        ax.bar(dts, series[p], bottom=bottom, color=colors[p], width=0.9,
               label=p if p != "other" else other_label)
        bottom += series[p]
    style_axis(ax)
    _date_axis(ax, dts)
    ax.set_title("AI Token Usage — Daily (stacked by project)", loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, labelcolor="#e6edf3",
              fontsize=10, ncol=2)
    _save(fig, out_dir, "usage_daily_by_project.png")
    print(f"  {len(ranked)} projects total; top {top_n} shown")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _legend(ax, fams, colors):
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in fams]
    ax.legend(handles, fams, loc="upper left", frameon=False,
              labelcolor="#e6edf3", fontsize=11)


def _date_axis(ax, dts, minor=True):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    if minor:
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.set_xlim(dts[0] - timedelta(days=1), dts[-1] + timedelta(days=1))


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("charts", nargs="*",
                   choices=["monthly", "daily", "rolling", "projects",
                            "cost", "all"],
                   default=["all"],
                   help="which chart(s) to generate (default: all)")
    p.add_argument("--out-dir", default=os.path.join(here, "charts"),
                   help="output directory for PNGs (default: ./charts)")
    p.add_argument("--claude-dir", default=os.path.expanduser("~/.claude/projects"),
                   help="Claude Code projects log directory")
    p.add_argument("--codex-dir", default=os.path.expanduser("~/.codex/sessions"),
                   help="Codex sessions log directory (use '' to skip Codex)")
    args = p.parse_args()

    wanted = set(args.charts)
    if "all" in wanted:
        wanted = {"monthly", "daily", "rolling", "projects", "cost"}

    records = collect_usage(args.claude_dir, args.codex_dir or "")
    total = sum(r["tokens"] for r in records)
    cost = sum(record_cost(r) for r in records)
    print(f"parsed {len(records):,} usage records, "
          f"{fmt_tokens(total)} tokens, ~{fmt_dollars(cost)} est. cost")

    apply_theme()
    if "monthly" in wanted:
        chart_monthly(records, args.out_dir)
    if "daily" in wanted:
        chart_daily(records, args.out_dir)
    if "rolling" in wanted:
        chart_rolling(records, args.out_dir)
    if "projects" in wanted:
        chart_projects(records, args.out_dir)
    if "cost" in wanted:
        chart_cost(records, args.out_dir)


if __name__ == "__main__":
    main()
