# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | dataset sanity | 2D Darcy generator smoke test | train-mini | shape, NaN, BC validity | MUST | TODO | validate manifest and tensor contracts |
| R002 | M0 | metric sanity | Darcy residual metric check | train-mini | residual correctness | MUST | TODO | compare against analytic or finite-difference reference |
| R003 | M0 | repair sanity | flux repair forward stability | train-mini | conservation, NaN | MUST | TODO | no training, operator-only validation |
| R004 | M0 | overfit sanity | residual backbone one-batch overfit | tiny split | Rel-L2, conservation | MUST | TODO | ensure pipeline can learn |
| R005 | M1 | anchor baseline | Coarse interpolation only | val | Rel-L2, mass error, flux mismatch | MUST | TODO | lower bound baseline |
| R006 | M1 | anchor baseline | DirectNO seed 1 | val | Rel-L2, mass error, flux mismatch | MUST | TODO | matched backbone |
| R007 | M1 | anchor baseline | DirectNO seed 2 | val | same | MUST | TODO |  |
| R008 | M1 | anchor baseline | DirectNO seed 3 | val | same | MUST | TODO |  |
| R009 | M1 | anchor baseline | ResidualNO seed 1 | val | same | MUST | TODO | residual-only |
| R010 | M1 | anchor baseline | ResidualNO seeds 2-3 | val | same | MUST | TODO | collapse if variance small |
| R011 | M2 | main method | ResidualNO+FluxRepair seed 1 | val | same | MUST | TODO | first decisive run |
| R012 | M2 | main method | ResidualNO+FluxRepair seed 2 | val | same | MUST | TODO |  |
| R013 | M2 | main method | ResidualNO+FluxRepair seed 3 | val | same | MUST | TODO |  |
| R014 | M3 | novelty isolation | ResidualNO+SoftPDE seed 1 | val | same | MUST | TODO | compare to repair |
| R015 | M3 | novelty isolation | ResidualNO+SoftPDE seeds 2-3 | val | same | MUST | TODO |  |
| R016 | M3 | novelty isolation | ResidualNO+SoftPDE+SoftFlux seed 1 | val | same | MUST | TODO | strongest soft baseline |
| R017 | M3 | novelty isolation | ResidualNO+SoftPDE+SoftFlux seeds 2-3 | val | same | MUST | TODO |  |
| R018 | M3 | novelty isolation | ResidualNO+BoundaryClamp | val | same | MUST | TODO | isolates BC handling |
| R019 | M3 | novelty isolation | before-vs-after repair analysis | val | repair delta, conservation delta | MUST | TODO | explicit mechanism evidence |
| R020 | M3 | novelty isolation | worst-cell / tail statistics | val | percentile conservation errors | MUST | TODO | reviewer-facing analysis |
| R021 | M4 | robustness | SoftPhys baseline on higher contrast | test-ood-contrast | same | MUST | TODO | OOD contrast sweep |
| R022 | M4 | robustness | FluxRepair on higher contrast | test-ood-contrast | same | MUST | TODO | compare degradation slope |
| R023 | M4 | robustness | DirectNO on higher contrast | test-ood-contrast | same | MUST | TODO | reference point |
| R024 | M4 | robustness | SoftPhys on BC variation | test-ood-bc | same | MUST | TODO | varying BC family |
| R025 | M4 | robustness | FluxRepair on BC variation | test-ood-bc | same | MUST | TODO | key robustness run |
| R026 | M4 | robustness | qualitative OOD case collection | test-ood | maps, failure cases | MUST | TODO | save paper figures |
| R027 | M5 | simplicity | FluxRepair + larger backbone | val | same + params | NICE | TODO | overbuilt comparison |
| R028 | M5 | simplicity | FluxRepair + extra head | val | same + params | NICE | TODO | complexity defense |
| R029 | M5 | appendix | extra seed sweep for top 2 methods | val | mean/std | NICE | TODO | if variance concern remains |
| R030 | M5 | appendix | fine-grid size sensitivity | alt resolution | same | NICE | TODO | optional scale check |
| R031 | M5 | appendix | qualitative figure panel | selected test | plots | NICE | TODO | publication assets |
| R032 | M5 | appendix | failure diagnostics write-up | worst cases | maps, notes | NICE | TODO | discussion support |
