# agentic-ai-usage

Generates charts summarizing AI coding-agent token usage, parsed **directly
from local agent logs** — no external tools or network access required.

| Agent | Log location parsed |
|-------|---------------------|
| Claude Code | `~/.claude/projects/**/*.jsonl` (incl. nested subagent logs) |
| Codex | `~/.codex/sessions/**/rollout-*.jsonl` |

Most charts plot **time on the X axis, tokens on the Y axis**; the `cost` chart
plots **estimated USD** instead. Every chart kind is rendered at two
granularities — **monthly** and **daily** — into separate PNGs.

## Usage

```bash
pixi run all            # generate every chart into ./charts
pixi run models         # tokens, stacked by model version
pixi run tokentype      # tokens, stacked by token type
pixi run worktokens     # tokens by token type, excluding cache reads
pixi run context        # avg cache reads per request, Claude (context proxy)
pixi run cachehit       # cache hit ratio (cache reads / input-side tokens)
pixi run cost           # estimated cost (USD), stacked by token type
pixi run reasoning      # Codex reasoning vs visible output tokens
pixi run projects       # tokens, stacked by project (top 11 + "other")
pixi run prompts        # count of input prompts, stacked by project
pixi run interruptions  # interrupts per prompt (steering-friction rate)
pixi run turns          # model turns per prompt (agentic leverage)
pixi run tokensperprompt  # avg tokens per prompt, by project (single chart)
pixi run gantt          # project activity timeline (single chart)
```

Most kinds are **periodic** — they write `usage_<kind>_monthly.png` and
`usage_<kind>_daily.png`. The two single charts (`tokensperprompt`, `gantt`)
write one `usage_<kind>.png`.

| Chart kind | Files |
|------------|-------|
| `models`        | `usage_models_{monthly,daily}.png` |
| `tokentype`     | `usage_tokentype_{monthly,daily}.png` |
| `worktokens`    | `usage_worktokens_{monthly,daily}.png` |
| `context`       | `usage_context_{monthly,daily}.png` |
| `cachehit`      | `usage_cachehit_{monthly,daily}.png` |
| `cost`          | `usage_cost_{monthly,daily}.png` |
| `reasoning`     | `usage_reasoning_{monthly,daily}.png` |
| `projects`      | `usage_projects_{monthly,daily}.png` |
| `prompts`       | `usage_prompts_{monthly,daily}.png` |
| `interruptions` | `usage_interruptions_{monthly,daily}.png` |
| `turns`         | `usage_turns_{monthly,daily}.png` |
| `tokensperprompt` | `usage_tokensperprompt.png` |
| `gantt`         | `usage_gantt.png` |

### Options

Run the script directly for more control:

```bash
pixi run python analyze.py models projects --out-dir /tmp/charts
pixi run python analyze.py all --claude-dir ~/.claude/projects --codex-dir ''
```

- `--out-dir`     output directory for PNGs (default: `./charts`)
- `--claude-dir`  Claude Code log directory
- `--codex-dir`   Codex log directory (pass `''` to skip Codex)

## Notes

- Token counts include cache reads/writes, which dominate the totals. Because
  cache reads are billed at ~0.1x the input rate, the token charts overstate
  cost differences — use the `cost` chart for a cost-representative view.
- The model charts stack by specific version (e.g. `Opus 4.7`, `Haiku 4.5`),
  parsed from the logged model id and shaded within each family's color (older
  = darker, newer = lighter). Codex logs carry no model id, so they show as a
  single "Codex"; pricing is still applied per family.
- Codex usage is summed from per-turn `last_token_usage` deltas to avoid
  double-counting the cumulative `total_token_usage`.
- The `prompts` chart counts the messages you actually typed: non-meta user
  messages (excluding tool results), deduplicated by `promptId` so one prompt
  isn't counted per fan-out turn, plus Codex `user_message` events. Slash
  commands count as prompts; older Claude logs without `promptId` fall back to
  per-message counting.
- `turns` is *agentic leverage*: model turns (API responses) ÷ prompts — how
  much autonomous work each instruction triggers. `interruptions` is a *rate*
  — `[Request interrupted by user]` messages plus Codex `turn_aborted` events,
  divided by prompts — so it reflects steering friction, not raw volume.
  `reasoning` shows Codex `reasoning_output_tokens` vs visible output (Codex
  only; Claude doesn't report reasoning separately).

### Cost estimation

The `cost` chart weights each token component by price rather than counting
tokens equally, and stacks the result by **token type** (input / output / cache
write / cache read) so you can see what's actually driving spend. Per-family
list prices (USD per million tokens) live in the `PRICES` table in `analyze.py`;
cache writes are billed at 1.25x input and cache reads at 0.10x input. **Edit
`PRICES` to match your actual rates** — the estimate is only as accurate as that
table, and it ignores tier/discount deals.
