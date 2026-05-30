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
import matplotlib.colors as mcolors
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

# Token-type breakdown shared by the token-type and cost charts.
COST_COMPONENTS = ["Input", "Output", "Cache write", "Cache read"]
COMPONENT_COLORS = {"Input": "#6e8fd4", "Output": "#d97757",
                    "Cache write": "#e0b341", "Cache read": "#7ec699"}
COMPONENT_FIELD = {"Input": "input", "Output": "output",
                   "Cache write": "cache_create", "Cache read": "cache_read"}


def token_components(r):
    """Raw token counts of one usage record, split by token type."""
    return {label: r[field] for label, field in COMPONENT_FIELD.items()}


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


def fmt_count(x, _=None):
    return f"{x:,.0f}"


def fmt_ratio(x, _=None):
    return f"{x:.1f}"


def fmt_pct(x, _=None):
    return f"{x:.0%}"


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
    return "Other"  # <synthetic> etc. (Codex is tagged directly in parse_codex)


def model_label(name):
    """Family + version, e.g. 'claude-opus-4-7' -> 'Opus 4.7'.

    Falls back to the bare family when no version is encoded in the id.
    """
    fam = model_family(name)
    if fam == "Other":
        return "Other"
    parts = name.split("-")  # claude-<family>-<major>-<minor>[-date]
    if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
        return f"{fam} {parts[2]}.{parts[3]}"
    return fam


def parse_claude(claude_dir):
    """Yield usage records from Claude Code logs (recursive; deduplicated).

    A streamed response logs the same (message.id, requestId) several times
    with growing `output_tokens`; only the last emission holds the complete
    usage. We therefore keep the record with the largest output per key rather
    than the first one seen.
    """
    best = {}  # key -> (output_tokens, record)
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
            inp = u.get("input_tokens", 0)
            out = u.get("output_tokens", 0)
            cc = u.get("cache_creation_input_tokens", 0)
            cr = u.get("cache_read_input_tokens", 0)
            tok = inp + out + cc + cr
            if tok <= 0:
                continue
            mid, rid = msg.get("id"), o.get("requestId")
            key = (mid, rid) if (mid and rid) else o.get("uuid")
            if key in best and out <= best[key][0]:
                continue
            cwd = o.get("cwd")
            name = msg.get("model", "")
            best[key] = (out, {
                "date": ts[:10],
                "project": os.path.basename(cwd) if cwd else "unknown",
                "family": model_family(name),
                "model": model_label(name),
                "tokens": tok,
                "input": inp, "output": out,
                "cache_create": cc, "cache_read": cr,
                "reasoning": 0,  # Claude doesn't report reasoning separately
            })
    for _, rec in best.values():
        yield rec


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
                "model": "Codex",  # Codex logs carry no model id
                "tokens": tok,
                "input": max(last.get("input_tokens", 0) - cached, 0),
                "output": last.get("output_tokens", 0),
                "cache_create": 0,
                "cache_read": cached,
                "reasoning": last.get("reasoning_output_tokens", 0),
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
# prompt parsing  ->  one {date, project} per user prompt
# ----------------------------------------------------------------------------
def parse_claude_prompts(claude_dir):
    """Yield {date, project} for each user prompt.

    Counts non-meta user messages, skipping tool results (which are logged as
    user messages too). Deduplicates by promptId — one prompt fans out into
    several records — falling back to uuid for older logs that predate it.
    """
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
            if o.get("type") != "user" or o.get("isMeta"):
                continue
            content = (o.get("message") or {}).get("content")
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content):
                continue
            ts = o.get("timestamp")
            if not ts:
                continue
            key = o.get("promptId") or o.get("uuid")
            if key in seen:
                continue
            seen.add(key)
            cwd = o.get("cwd")
            yield {"date": ts[:10],
                   "project": os.path.basename(cwd) if cwd else "unknown"}


def parse_codex_prompts(codex_dir):
    """Yield {date, project} for each Codex user_message event."""
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
            ts = o.get("timestamp")
            if payload.get("type") != "user_message" or not ts:
                continue
            yield {"date": ts[:10],
                   "project": os.path.basename(cwd) if cwd else "unknown"}


def collect_prompts(claude_dir, codex_dir):
    prompts = []
    if os.path.isdir(claude_dir):
        prompts.extend(parse_claude_prompts(claude_dir))
    if codex_dir and os.path.isdir(codex_dir):
        prompts.extend(parse_codex_prompts(codex_dir))
    return prompts


def parse_claude_interruptions(claude_dir):
    """Yield {date, project} for each '[Request interrupted by user]' message."""
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
            if o.get("type") != "user" or o.get("isMeta"):
                continue
            content = (o.get("message") or {}).get("content")
            ts = o.get("timestamp")
            if not isinstance(content, list) or not ts:
                continue
            if any(isinstance(b, dict) and b.get("type") == "text"
                   and str(b.get("text", "")).startswith("[Request interrupted")
                   for b in content):
                cwd = o.get("cwd")
                yield {"date": ts[:10],
                       "project": os.path.basename(cwd) if cwd else "unknown"}


def parse_codex_interruptions(codex_dir):
    """Yield {date, project} for each Codex turn_aborted event."""
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
            ts = o.get("timestamp")
            if payload.get("type") != "turn_aborted" or not ts:
                continue
            yield {"date": ts[:10],
                   "project": os.path.basename(cwd) if cwd else "unknown"}


def collect_interruptions(claude_dir, codex_dir):
    rows = []
    if os.path.isdir(claude_dir):
        rows.extend(parse_claude_interruptions(claude_dir))
    if codex_dir and os.path.isdir(codex_dir):
        rows.extend(parse_codex_interruptions(codex_dir))
    return rows


# ----------------------------------------------------------------------------
# aggregation helpers
# ----------------------------------------------------------------------------
def by_period_group(records, key, group="model"):
    """period -> {value: tokens}. key(date) selects month ('%Y-%m') or day;
    group selects the field to stack by ('model' or 'family')."""
    agg = defaultdict(lambda: defaultdict(int))
    for r in records:
        agg[key(r["date"])][r[group]] += r["tokens"]
    return agg


def _version_key(label):
    """Sort key for a model label like 'Opus 4.7' -> 4.7 (family-less -> 0)."""
    tail = label.rsplit(" ", 1)
    try:
        return float(tail[1])
    except (IndexError, ValueError):
        return 0.0


def _shades(base, n):
    """n colors from one base hue; ordered older/darker -> newer/lighter."""
    rgb = np.array(mcolors.to_rgb(base))
    if n == 1:
        return [base]
    out = []
    for f in np.linspace(-0.30, 0.45, n):
        c = rgb + (1 - rgb) * f if f >= 0 else rgb * (1 + f)
        out.append(mcolors.to_hex(np.clip(c, 0, 1)))
    return out


def model_order_and_colors(records):
    """Stacking order (family order, then version ascending) and a color per
    model, shaded within each family's base hue."""
    by_fam = defaultdict(set)
    for r in records:
        by_fam[r["family"]].add(r["model"])
    order, colors = [], {}
    for fam in FAMILIES + [f for f in by_fam if f not in FAMILIES]:
        if fam not in by_fam:
            continue
        models = sorted(by_fam[fam], key=_version_key)
        for m, c in zip(models, _shades(MODEL_COLORS.get(fam, "#7e8a99"),
                                        len(models))):
            order.append(m)
            colors[m] = c
    return order, colors


def cost_by_period_component(records, key):
    """period -> {token-type: USD}. key(date) selects month ('%Y-%m') or day."""
    agg = defaultdict(lambda: defaultdict(float))
    for r in records:
        bucket = agg[key(r["date"])]
        for comp, usd in cost_components(r).items():
            bucket[comp] += usd
    return agg


def tokens_by_period_component(records, key):
    """period -> {token-type: tokens}. key(date) selects month or day."""
    agg = defaultdict(lambda: defaultdict(int))
    for r in records:
        bucket = agg[key(r["date"])]
        for comp, n in token_components(r).items():
            bucket[comp] += n
    return agg


# ----------------------------------------------------------------------------
# charts  (each kind renders for a period: "monthly" or "daily")
# ----------------------------------------------------------------------------
def _keyfn(period):
    """date string -> bucket key: month ('%Y-%m') for monthly, full day else."""
    return (lambda d: d[:7]) if period == "monthly" else (lambda d: d)


def _bar_x(keys, period):
    """x positions and bar width: categorical months or datetime days."""
    if period == "monthly":
        return list(keys), 0.62
    return [datetime.strptime(k, "%Y-%m-%d") for k in keys], 0.9


def _value_axis(ax, fmt, ylabel):
    ax.grid(axis="y", color="#30363d", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_ylabel(ylabel)


def _bar_over_time(keys, vals, period, *, out_dir, fname, title, fmt, ylabel,
                   color=PALETTE[1]):
    """Render and save a single-series bar chart over time."""
    if not keys:
        return
    x, width = _bar_x(keys, period)
    fig, ax = plt.subplots(figsize=(10, 5.6) if period == "monthly"
                           else (13, 5.6), dpi=160)
    ax.bar(x, vals, color=color, width=width)
    if period == "monthly":
        for i, v in enumerate(vals):
            ax.text(i, v, " " + fmt(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#e6edf3")
        ax.set_ylim(0, (max(vals) or 1) * 1.12)
    else:
        _date_axis(ax, x)
    _value_axis(ax, fmt, ylabel)
    ax.set_title(title, loc="left", pad=14)
    _save(fig, out_dir, fname)


def _draw_stacked(agg, order, colors, period, *, out_dir, fname, title,
                  fmt=fmt_tokens, ylabel="Tokens", labels=None, ncol=1):
    """Render and save one stacked-bar chart for the given period."""
    keys = sorted(agg)
    if not keys:
        return
    x, width = _bar_x(keys, period)
    fig, ax = plt.subplots(figsize=(10, 5.6) if period == "monthly"
                           else (13, 5.6), dpi=160)
    label_of = (lambda n: labels.get(n, n)) if labels else (lambda n: n)
    bottom = np.zeros(len(keys))
    used = []
    for name in order:
        vals = np.array([agg[k].get(name, 0) for k in keys])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, color=colors[name], width=width,
               label=label_of(name))
        bottom += vals
        used.append(name)
    if period == "monthly":
        for i, t in enumerate(bottom):
            ax.text(i, t, " " + fmt(t), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#e6edf3")
        ax.set_ylim(0, (bottom.max() or 1) * 1.12)
    else:
        _date_axis(ax, x)
    _value_axis(ax, fmt, ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[n]) for n in used]
    ax.legend(handles, [label_of(n) for n in used], loc="upper left",
              frameon=False, labelcolor="#e6edf3",
              fontsize=11 if ncol == 1 else 10, ncol=ncol)
    ax.set_title(title, loc="left", pad=14)
    _save(fig, out_dir, fname)


def chart_models(records, out_dir, period):
    order, colors = model_order_and_colors(records)
    agg = by_period_group(records, _keyfn(period))
    _draw_stacked(agg, order, colors, period, out_dir=out_dir,
                  fname=f"usage_models_{period}.png",
                  title=f"AI Token Usage — {period.title()} (stacked by model)")


def chart_tokentype(records, out_dir, period, *, exclude_cache_read=False):
    agg = tokens_by_period_component(records, _keyfn(period))
    order = [c for c in COST_COMPONENTS
             if not (exclude_cache_read and c == "Cache read")]
    kind = "worktokens" if exclude_cache_read else "tokentype"
    extra = ", excl. cache reads" if exclude_cache_read else ""
    _draw_stacked(agg, order, COMPONENT_COLORS, period, out_dir=out_dir,
                  fname=f"usage_{kind}_{period}.png",
                  title=f"AI Token Usage — {period.title()} "
                        f"(by token type{extra})")


def chart_cost(records, out_dir, period):
    agg = cost_by_period_component(records, _keyfn(period))
    _draw_stacked(agg, COST_COMPONENTS, COMPONENT_COLORS, period,
                  out_dir=out_dir, fname=f"usage_cost_{period}.png",
                  title=f"AI Cost — {period.title()} est. (by token type)",
                  fmt=fmt_dollars, ylabel="Estimated cost (USD)")


def chart_context(records, out_dir, period):
    """Avg cache-read tokens per request — a proxy for context size per turn.

    Cache reads dominate any agentic session, so their *per-request* size is
    the real signal of how much context each turn was replaying. Restricted to
    Claude, whose caching is comparable across requests (Codex logs report
    caching differently and would dilute the average).
    """
    keyfn = _keyfn(period)
    cr = defaultdict(int)
    n = defaultdict(int)
    for r in records:
        if r["family"] == "Codex":
            continue
        k = keyfn(r["date"])
        cr[k] += r["cache_read"]
        n[k] += 1
    keys = sorted(cr)
    vals = [cr[k] / n[k] for k in keys]
    _bar_over_time(keys, vals, period, out_dir=out_dir,
                   fname=f"usage_context_{period}.png", fmt=fmt_tokens,
                   ylabel="Avg cache-read tokens / request",
                   color=COMPONENT_COLORS["Cache read"],
                   title=f"Context per Request — {period.title()} "
                         f"(avg cache reads / request, Claude only)")


def _by_project_chart(records, out_dir, period, *, kind, value, title,
                      fmt=fmt_tokens, ylabel="Tokens", top_n=11):
    """Stacked-by-project chart; `value(r)` is the per-record quantity summed."""
    keyfn = _keyfn(period)
    agg = defaultdict(lambda: defaultdict(float))
    proj_total = defaultdict(float)
    for r in records:
        v = value(r)
        agg[keyfn(r["date"])][r["project"]] += v
        proj_total[r["project"]] += v
    ranked = sorted(proj_total.items(), key=lambda x: -x[1])
    top = {p for p, _ in ranked[:top_n]}
    has_other = len(ranked) > top_n
    collapsed = defaultdict(lambda: defaultdict(float))
    for k, projs in agg.items():
        for proj, v in projs.items():
            collapsed[k][proj if proj in top else "other"] += v
    order = [p for p, _ in ranked[:top_n]] + (["other"] if has_other else [])
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(order)}
    labels = {}
    if has_other:
        colors["other"] = "#5a6473"
        labels["other"] = f"other ({len(ranked) - top_n} projects)"
    _draw_stacked(collapsed, order, colors, period, out_dir=out_dir,
                  fname=f"usage_{kind}_{period}.png", title=title,
                  fmt=fmt, ylabel=ylabel, labels=labels, ncol=2)
    if period == "monthly":
        print(f"  {len(ranked)} projects total; top {top_n} shown")


def chart_projects(records, out_dir, period):
    _by_project_chart(records, out_dir, period, kind="projects",
                      value=lambda r: r["tokens"],
                      title=f"AI Token Usage — {period.title()} "
                            f"(stacked by project)")


def chart_prompts(records, out_dir, period):
    _by_project_chart(records, out_dir, period, kind="prompts",
                      value=lambda r: 1, fmt=fmt_count, ylabel="Prompts",
                      title=f"Input Prompts — {period.title()} "
                            f"(stacked by project)")


def chart_interruptions(data, out_dir, period):
    """Interruptions per prompt over time — steering friction, volume-normalized.

    A raw count just tracks usage; dividing by prompts shows how often a typical
    instruction had to be course-corrected.
    """
    keyfn = _keyfn(period)
    ints = defaultdict(int)
    proms = defaultdict(int)
    for r in data["interruptions"]:
        ints[keyfn(r["date"])] += 1
    for r in data["prompts"]:
        proms[keyfn(r["date"])] += 1
    keys = [k for k in sorted(proms) if proms[k]]
    vals = [ints[k] / proms[k] for k in keys]
    _bar_over_time(keys, vals, period, out_dir=out_dir,
                   fname=f"usage_interruptions_{period}.png", fmt=fmt_pct,
                   ylabel="Interruptions / prompt", color="#e0739b",
                   title=f"Interruption Rate — {period.title()} "
                         f"(interrupts per prompt)")


def chart_turns(data, out_dir, period):
    """Model turns per prompt over time — a measure of agentic leverage."""
    keyfn = _keyfn(period)
    turns = defaultdict(int)
    proms = defaultdict(int)
    for r in data["usage"]:
        turns[keyfn(r["date"])] += 1
    for r in data["prompts"]:
        proms[keyfn(r["date"])] += 1
    keys = [k for k in sorted(set(turns) | set(proms)) if proms.get(k)]
    vals = [turns[k] / proms[k] for k in keys]
    _bar_over_time(keys, vals, period, out_dir=out_dir,
                   fname=f"usage_turns_{period}.png", fmt=fmt_ratio,
                   ylabel="Model turns / prompt", color=PALETTE[4],
                   title=f"Agentic Leverage — {period.title()} "
                         f"(model turns per prompt)")


def chart_cachehit(data, out_dir, period):
    """Share of input-side tokens served from cache, over time."""
    keyfn = _keyfn(period)
    hit = defaultdict(int)
    tot = defaultdict(int)
    for r in data["usage"]:
        k = keyfn(r["date"])
        hit[k] += r["cache_read"]
        tot[k] += r["input"] + r["cache_create"] + r["cache_read"]
    keys = [k for k in sorted(tot) if tot[k]]
    vals = [hit[k] / tot[k] for k in keys]
    _bar_over_time(keys, vals, period, out_dir=out_dir,
                   fname=f"usage_cachehit_{period}.png", fmt=fmt_pct,
                   ylabel="Cache hit ratio", color=COMPONENT_COLORS["Cache read"],
                   title=f"Cache Hit Ratio — {period.title()} "
                         f"(cache reads / input-side tokens)")


def chart_reasoning(data, out_dir, period):
    """Codex reasoning vs visible output tokens over time (Codex only)."""
    keyfn = _keyfn(period)
    agg = defaultdict(lambda: defaultdict(int))
    for r in data["usage"]:
        if r["family"] != "Codex":
            continue
        k = keyfn(r["date"])
        agg[k]["Reasoning"] += r["reasoning"]
        agg[k]["Output (visible)"] += max(r["output"] - r["reasoning"], 0)
    colors = {"Reasoning": "#b083d9", "Output (visible)": "#d97757"}
    _draw_stacked(agg, ["Reasoning", "Output (visible)"], colors, period,
                  out_dir=out_dir, fname=f"usage_reasoning_{period}.png",
                  title=f"Codex Output Tokens — {period.title()} "
                        f"(reasoning vs visible)")


def chart_tokens_per_prompt(data, out_dir, _period=None, top_n=11):
    """Avg tokens per prompt for the top projects (single all-time ranking)."""
    tok = defaultdict(int)
    pr = defaultdict(int)
    for r in data["usage"]:
        tok[r["project"]] += r["tokens"]
    for r in data["prompts"]:
        pr[r["project"]] += 1
    ranked = [p for p in sorted(tok, key=lambda p: -tok[p]) if pr.get(p)][:top_n]
    pairs = sorted(((p, tok[p] / pr[p]) for p in ranked), key=lambda x: x[1])
    if not pairs:
        return
    names = [p for p, _ in pairs]
    vals = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) + 1.8), dpi=160)
    ax.barh(names, vals,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(names))])
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt_tokens(v), va="center", ha="left",
                fontsize=10, fontweight="bold", color="#e6edf3")
    ax.grid(axis="x", color="#30363d", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_xlabel("Tokens / prompt")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title(f"Tokens per Prompt — by project (top {top_n} by total tokens)",
                 loc="left", pad=14)
    _save(fig, out_dir, "usage_tokensperprompt.png")


def chart_gantt(data, out_dir, _period=None, top_n=20):
    """Project activity timeline: first→last day of usage per project."""
    first, last, tok = {}, {}, defaultdict(int)
    for r in data["usage"]:
        p, d = r["project"], r["date"]
        first[p] = min(first.get(p, d), d)
        last[p] = max(last.get(p, d), d)
        tok[p] += r["tokens"]
    projs = sorted(tok, key=lambda p: -tok[p])[:top_n]
    projs.sort(key=lambda p: first[p], reverse=True)  # earliest at top
    if not projs:
        return
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(projs) + 1.6), dpi=160)
    for i, p in enumerate(projs):
        x0 = mdates.date2num(datetime.strptime(first[p], "%Y-%m-%d"))
        x1 = mdates.date2num(datetime.strptime(last[p], "%Y-%m-%d"))
        ax.barh(i, x1 - x0 + 1, left=x0, height=0.6,
                color=PALETTE[i % len(PALETTE)])
    ax.set_yticks(range(len(projs)))
    ax.set_yticklabels(projs, fontsize=9)
    ax.set_ylim(-0.6, len(projs) - 0.4)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color="#30363d", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"Project Timeline (top {top_n} by tokens)", loc="left", pad=14)
    _save(fig, out_dir, "usage_gantt.png")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
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
# kind -> spec. fn(data, out_dir, period); `periodic` kinds render monthly +
# daily, others render once (period=None); `needs` lists extra datasets to load.
CHART_KINDS = {
    "models":      dict(fn=lambda d, o, p: chart_models(d["usage"], o, p),
                        periodic=True, needs=()),
    "tokentype":   dict(fn=lambda d, o, p: chart_tokentype(d["usage"], o, p),
                        periodic=True, needs=()),
    "worktokens":  dict(fn=lambda d, o, p: chart_tokentype(
                        d["usage"], o, p, exclude_cache_read=True),
                        periodic=True, needs=()),
    "context":     dict(fn=lambda d, o, p: chart_context(d["usage"], o, p),
                        periodic=True, needs=()),
    "cachehit":    dict(fn=chart_cachehit, periodic=True, needs=()),
    "cost":        dict(fn=lambda d, o, p: chart_cost(d["usage"], o, p),
                        periodic=True, needs=()),
    "reasoning":   dict(fn=chart_reasoning, periodic=True, needs=()),
    "projects":    dict(fn=lambda d, o, p: chart_projects(d["usage"], o, p),
                        periodic=True, needs=()),
    "prompts":     dict(fn=lambda d, o, p: chart_prompts(d["prompts"], o, p),
                        periodic=True, needs=("prompts",)),
    "interruptions": dict(fn=chart_interruptions, periodic=True,
                          needs=("prompts", "interruptions")),
    "turns":       dict(fn=chart_turns, periodic=True, needs=("prompts",)),
    "tokensperprompt": dict(fn=chart_tokens_per_prompt, periodic=False,
                            needs=("prompts",)),
    "gantt":       dict(fn=chart_gantt, periodic=False, needs=()),
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("charts", nargs="*",
                   choices=list(CHART_KINDS) + ["all"],
                   default=["all"],
                   help="chart kind(s) to generate (default: all); periodic "
                        "kinds produce a monthly and a daily PNG")
    p.add_argument("--out-dir", default=os.path.join(here, "charts"),
                   help="output directory for PNGs (default: ./charts)")
    p.add_argument("--claude-dir", default=os.path.expanduser("~/.claude/projects"),
                   help="Claude Code projects log directory")
    p.add_argument("--codex-dir", default=os.path.expanduser("~/.codex/sessions"),
                   help="Codex sessions log directory (use '' to skip Codex)")
    args = p.parse_args()

    sel = set(args.charts)
    kinds = list(CHART_KINDS) if "all" in sel else \
        [k for k in CHART_KINDS if k in sel]

    records = collect_usage(args.claude_dir, args.codex_dir or "")
    total = sum(r["tokens"] for r in records)
    cost = sum(record_cost(r) for r in records)
    print(f"parsed {len(records):,} usage records, "
          f"{fmt_tokens(total)} tokens, ~{fmt_dollars(cost)} est. cost")

    needs = set().union(*(CHART_KINDS[k]["needs"] for k in kinds)) if kinds \
        else set()
    data = {"usage": records}
    if "prompts" in needs:
        data["prompts"] = collect_prompts(args.claude_dir, args.codex_dir or "")
        print(f"parsed {len(data['prompts']):,} prompts")
    if "interruptions" in needs:
        data["interruptions"] = collect_interruptions(args.claude_dir,
                                                       args.codex_dir or "")
        print(f"parsed {len(data['interruptions']):,} interruptions")

    apply_theme()
    for kind in kinds:
        spec = CHART_KINDS[kind]
        for period in (("monthly", "daily") if spec["periodic"] else (None,)):
            spec["fn"](data, args.out_dir, period)


if __name__ == "__main__":
    main()
