# WORKTREE PLAN

## Goal

Use isolated Git worktrees so multiple agents can work in parallel without stomping on each other.

## Recommended layout

Create worktrees as sibling directories to the main repository:

```text
D:\Download\
  refine-logs\
  refine-logs-wt-a1-sanity\
  refine-logs-wt-a2-baseline\
  refine-logs-wt-a3-pcr\
  refine-logs-wt-a6-metrics\
```

## Recommended branch names

- `feat/a1-s0-sanity`
- `feat/a2-s1-baseline`
- `feat/a3-s2-pcr-core`
- `feat/a4-s3-ablation`
- `feat/a5-s4-robustness`
- `feat/a6-metrics-vis`
- `feat/a7-writing`

## Initial rollout

Start with four worktrees only:

1. A1 sanity
2. A2 baseline
3. A3 PCR core
4. A6 metrics

Delay A4 and A5 until `S2_best.pt` exists.

## Ownership by worktree

| Worktree | Branch | Agent | Primary write scope |
|---|---|---|---|
| `refine-logs-wt-a1-sanity` | `feat/a1-s0-sanity` | A1 | dataset contract, sanity script |
| `refine-logs-wt-a2-baseline` | `feat/a2-s1-baseline` | A2 | baseline script, baseline config |
| `refine-logs-wt-a3-pcr` | `feat/a3-s2-pcr-core` | A3 | PCR model and main trainer |
| `refine-logs-wt-a6-metrics` | `feat/a6-metrics-vis` | A6 | metric utilities, plotting, report templates |

## Commands

Run from the main repository root:

```bash
git worktree add ..\refine-logs-wt-a1-sanity -b feat/a1-s0-sanity
git worktree add ..\refine-logs-wt-a2-baseline -b feat/a2-s1-baseline
git worktree add ..\refine-logs-wt-a3-pcr -b feat/a3-s2-pcr-core
git worktree add ..\refine-logs-wt-a6-metrics -b feat/a6-metrics-vis
```

## Merge policy

- merge A1 before A2 depends on it
- A2 and A6 can merge independently
- A3 rebases or merges main after A2 lands
- A4 and A5 branch only after A3 checkpoint logic is stable

## Notes

- each agent should commit only within its declared write scope
- use `AGENT_HANDOFFS.md` for state transfer, not ad hoc chat summaries
- close and prune finished worktrees after merge
