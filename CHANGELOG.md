# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- TBD

## [0.1.1] - 2025-11-29

### Fixed
- Removed unused `map_patient_to_han2012` function from `cox_model.py`
- Fixed all PEP8/ruff linting violations (whitespace, import order)
- Updated ruff configuration to use non-deprecated `[tool.ruff.lint]` section
- Completed type annotations for all public functions

### Added
- Pre-commit hooks for automated code quality enforcement
- GitHub Actions CI/CD pipeline with multi-Python-version testing
- Test coverage reporting with 70% minimum threshold
- SECURITY.md for responsible disclosure policy

### Changed
- Repository renamed to fix "calculater" typo

## [0.1.0] - 2025-11-26

### Added
- Han 2012 Cox proportional hazards survival model implementation
  - 5-year and 10-year overall survival predictions
  - Baseline survival calibrated to published cohort statistics
- KLASS-inspired heuristic recurrence model (educational demonstration)
- TCGA STAD cohort validation pipeline
  - Variable harmonization via `Han2012VariableMapper`
  - Stage-informed imputation for missing surgical variables
  - Brier score calibration analysis
- Sensitivity analysis for lymph node yield impact
- Publication-quality visualization suite
  - Calibration curves
  - Survival distribution histograms
  - Risk stratification heatmaps
- Docker containerization for reproducible execution
- Comprehensive test suite (13 tests, 100% pass rate)

### Documentation
- TRIPOD-compliant validation report
- Clinical translation roadmap (Phases 1-4)
- Pre-deployment checklist for institutional use
- Full mathematical specification of model coefficients

### Known Limitations
- Baseline survival S₀(t) estimated, not from original publication
- TCGA validation uses 100% imputed surgical variables
- Brier score reflects endpoint mismatch (recurrence vs. DFS), not model failure
