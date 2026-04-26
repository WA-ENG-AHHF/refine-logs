# DATASET CONTRACT

## Purpose

Freeze the dataset interface before baseline or main model training begins.

## Required Fields
- dataset name:
- PDE family:
- grid size:
- input variables:
- target variables:
- train split:
- validation split:
- test split:
- coupling parameter range:
- boundary condition types:
- interpolation method:
- normalization method:
- conservation quantity:

## Validation Checklist
- shapes verified
- no NaN or Inf
- coarse-to-fine interpolation verified
- boundary encoding verified
- gradient path for physics loss verified
- checkpoint save/load smoke test verified

## Sign-off
- prepared by:
- reviewed by:
- date:
