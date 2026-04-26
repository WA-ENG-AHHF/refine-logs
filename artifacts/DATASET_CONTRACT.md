# DATASET CONTRACT

## Purpose

Freeze the dataset interface before baseline or main model training begins, and make the S0 handoff verifiable by another worker without guessing missing assumptions.

## How To Use

1. Fill every field in `Dataset Identity`, `Data Layout`, and `Split Definition`.
2. Replace each bracketed placeholder with a concrete value, path, or `N/A`.
3. Keep path examples project-relative when possible so the S0 script config can mirror them.
4. Mark each acceptance item as `pass`, `fail`, or `not run`, and include evidence when relevant.

## Dataset Identity

| Field | Value | Notes |
|---|---|---|
| dataset name | `[e.g. ADR-2D-v1]` | Stable identifier used in configs and reports |
| dataset version | `[e.g. 2026-04-26]` | Date tag or semantic version |
| owner / source | `[team, paper, generator, or external source]` | Where the data came from |
| PDE family | `[e.g. advection-diffusion-reaction]` | Expand acronym on first use |
| task type | `[forecasting / operator learning / super-resolution / inverse]` | One primary task |
| license / usage notes | `[internal / open / restricted]` | Include any access constraints |

## Data Layout

| Field | Value | Notes |
|---|---|---|
| dataset root directory | `[relative or absolute path]` | Should match `dataset.root_dir` in sanity config if used |
| raw file format | `[npy / npz / h5 / pt / csv / other]` | One primary storage format |
| sample unit | `[what one row/item means]` | Example: one PDE rollout |
| num samples | `[integer]` | Total count before split |
| grid size | `[e.g. 64 x 64]` | Include spatial and temporal axes if relevant |
| resolution policy | `[native / interpolated / mixed]` | Note any coarse-to-fine processing |
| input variables | `[list]` | Include channel order |
| target variables | `[list]` | Include channel order |
| auxiliary variables | `[list or N/A]` | Coefficients, masks, coordinates, etc. |
| tensor shape convention | `[e.g. batch, time, x, y, channel]` | Canonical ordering used by loaders |
| dtype | `[e.g. float32]` | Expected model-side dtype |
| value range | `[min/max or expected bounds]` | Per variable if ranges differ |

## Split Definition

| Field | Value | Notes |
|---|---|---|
| split method | `[random / scenario / temporal / file-based]` | Explain grouping logic |
| train split | `[count or %]` | Must sum with val/test |
| validation split | `[count or %]` | Must sum with train/test |
| test split | `[count or %]` | Must sum with train/val |
| split seed | `[integer or N/A]` | Required if randomized |
| leakage guard | `[rule used to prevent overlap]` | Example: scenario-level split |

## Physics And Boundary Metadata

| Field | Value | Notes |
|---|---|---|
| coupling parameter range | `[value or interval]` | Include units if relevant |
| boundary condition types | `[Dirichlet / Neumann / periodic / mixed]` | Name all that occur |
| boundary encoding method | `[mask / ghost cell / token / other]` | What the model sees |
| interpolation method | `[bilinear / bicubic / spectral / none]` | Used for coarse-to-fine path |
| normalization method | `[z-score / min-max / none / other]` | State fit scope: train-only or global |
| conservation quantity | `[mass / energy / none / other]` | Primary physical invariant to track |
| known failure modes | `[brief list or N/A]` | NaN zones, unstable parameters, etc. |

## Sanity Config Mapping

Fill this section so another worker can translate the contract into `execution-repo/configs/S0_sanity.yaml` without inference.

| Config Key | Expected Value |
|---|---|
| `stage` | `[e.g. S0]` |
| `dataset.name` | `[copy from dataset name]` |
| `dataset.grid_size` | `[integer only]` |
| `dataset.num_samples` | `[integer only]` |
| `dataset.root_dir` | `[path or N/A]` |
| `dataset.required_paths` | `[list of required files/dirs or N/A]` |
| `output.report_json` | `[e.g. logs/S0_sanity_report.json]` |

## Acceptance Checklist

Mark status and attach evidence. Suggested evidence: command output, report path, or short note with date.

| Item | Status | Evidence / Notes |
|---|---|---|
| Required fields are filled and internally consistent | `[pass/fail/not run]` | `[note]` |
| Declared sample count matches on-disk content | `[pass/fail/not run]` | `[note]` |
| Declared split sizes sum correctly | `[pass/fail/not run]` | `[note]` |
| No NaN or Inf found in checked subset | `[pass/fail/not run]` | `[note]` |
| Tensor shapes match documented convention | `[pass/fail/not run]` | `[note]` |
| Coarse-to-fine interpolation path verified | `[pass/fail/not run]` | `[note]` |
| Boundary encoding matches contract description | `[pass/fail/not run]` | `[note]` |
| Gradient path for physics loss verified | `[pass/fail/not run]` | `[note]` |
| Checkpoint save/load smoke test verified | `[pass/fail/not run]` | `[note]` |
| `logs/S0_sanity_report.json` generated successfully | `[pass/fail/not run]` | `[note]` |

## Handoff Notes

- assumptions:
  - `[list any assumptions another worker must preserve]`
- blockers:
  - `[list open issues or N/A]`
- follow-up actions:
  - `[next concrete step for A2 or reviewer]`

## Sign-off

| Role | Name | Date | Comment |
|---|---|---|---|
| prepared by | `[name]` | `[YYYY-MM-DD]` | `[optional note]` |
| reviewed by | `[name]` | `[YYYY-MM-DD]` | `[optional note]` |
