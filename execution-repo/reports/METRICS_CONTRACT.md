# Metrics Contract

This document defines the canonical field names that `scripts/plot_metrics.py` expects when aggregating experiment outputs into a master summary CSV.

## Canonical Metrics

All metric sources should map to the following canonical columns:

| Canonical field | Meaning | Expected direction | Notes |
| --- | --- | --- | --- |
| `rel_l2` | Relative L2 error between prediction and reference. | Lower is better. | Dimensionless scalar. Recommended formula: `||pred - ref||_2 / ||ref||_2`. |
| `pde_residual` | PDE residual magnitude for the governing equation. | Lower is better. | Use a scalar reduction such as mean, RMS, or normalized integral, and document that choice beside the experiment. |
| `bc_violation` | Boundary-condition violation magnitude. | Lower is better. | Aggregate across all enforced boundaries with a single scalar reduction. |
| `conservation_error` | Error in a conservation law such as mass or energy. | Lower is better. | Use an absolute or relative scalar and record the chosen law in experiment metadata if needed. |

## Accepted Aliases

`plot_metrics.py` normalizes keys by lowercasing and converting spaces, hyphens, dots, and slashes to underscores. The following aliases are accepted:

| Canonical field | Accepted aliases |
| --- | --- |
| `rel_l2` | `rel-L2`, `relative_l2`, `relative-L2`, `relative_l2_error`, `relative_error_l2`, `l2_relative_error` |
| `pde_residual` | `PDE residual`, `pde-residual`, `residual_pde`, `equation_residual` |
| `bc_violation` | `BC violation`, `bc-violation`, `boundary_violation`, `boundary_condition_violation` |
| `conservation_error` | `conservation error`, `conservation-error`, `mass_conservation_error`, `energy_conservation_error` |

## Required Output Shape

The aggregated master summary CSV uses these columns:

`source_file`, `source_type`, `record_index`, `run_id`, `experiment`, `case_id`, `step`, `epoch`, `split`, `tag`, `rel_l2`, `pde_residual`, `bc_violation`, `conservation_error`

Metadata columns may be empty when the source file does not provide them, but the metric columns must always be present in the output header.

## Input Expectations

- CSV inputs should use one record per row.
- JSON inputs may be:
  - a single object containing metrics,
  - a list of metric objects,
  - an object with a `metrics` list,
  - an object with a `records` list.
- Nested JSON objects are flattened with underscore-separated keys before alias matching.

## Authoring Guidance

- Prefer writing canonical names directly in new experiment outputs.
- Keep each metric as a scalar summary value rather than a raw vector field.
- If a metric uses a nonstandard normalization or reduction, record that in the experiment report so downstream plots remain comparable.
