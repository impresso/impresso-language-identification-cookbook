# Release Notes - v2.1.0

**Release Date:** 2026-05-26
**Tag:** v2.1.0
**Status:** Pending

## Overview

This release adds support for radio/audio canonical records as a new canonical
input kind alongside existing newspaper page support. It also introduces
Hugging Face local-files-only (offline) model loading, OCRQA statistics
aggregation in newspaper statistics, ensemble decision-making improvements,
and new provider support for SUB and BNF. The release bundles a new
configuration file (`v2-0-3`) that enables `CANONICAL_INPUT_KIND=auto` for
mixed newspaper/radio collections.

## Language Identification

### New: Audio/Radio Canonical Input

- Added `_extract_text_from_audio_record()` to handle audio canonical records
  (`audios/NEWSPAPER-YEAR/*-audios.jsonl.bz2`) as a first-class input kind.
- `ImpressoLangidentSystems` now accepts `canonical_input_kind` parameter
  (`"pages"` or `"audios"`); `"auto"` is resolved at the Make level via
  `CANONICAL_INPUT_KIND=auto` in the config.
- The processing pipeline treats audio records and newspaper pages through the
  same LID/OCRQA path; the input format difference is handled transparently.

### New: Hugging Face Local-Files-Only Mode

- Added `local_files_only` parameter to `ImpressoLangidentSystems` and
  propagated to `impresso_langident_pipeline` and OCRQA pipeline initializers.
- `_ensure_local_files_only_supported()` validates that the installed pipeline
  version honors the flag and aborts with a clear error if not.
- Controlled via `LANGIDENT_LOCAL_FILES_ONLY_OPTION ?= --local-files-only` in
  config; useful for air-gapped or cache-only processing environments.

### Ensemble Decision Improvements

- `alphabetical_ratio` is now always present in output (defaulting to `None`
  when not computed), ensuring schema consistency.
- New fallback decision labels: `dominant-by-lowvote` and `dominant-lowvote`
  for cases where no LID predictions are available or vote counts are too low.
- Improved length-based fallback: dominant language is assigned for non-image
  items with text length > 0 when standard voting fails.
- Handles `null` `alphabetical_ratio` by treating it as `1.0` in ratio checks.
- Enhanced `_log_statistics()` output including decision type distribution and
  dominant language summary.

### Malformed Canonical Page Handling

- Added `_should_ignore_gn_spacing()` to detect and skip canonical pages where
  the `gn` (good-neighbour spacing) flag is systematically set, indicating a
  malformed OCR input.

### OCRQA Statistics Aggregation

- `NewspaperStatistics` now collects OCRQA scores per content item and produces
  an `ocrqa_statistics` summary including:
  - `items_with_ocrqa`, `coverage_ratio`
  - `avg_scores_per_language`, `dominant_language_by_ocrqa`
  - `agreement_with_ensemble` (OCRQA vs. ensemble decision agreement rate)
- `ocrqa_statistics` is `None` when no OCRQA data is present in the input.

### Provider Coverage

- **SUB** (Staats- und Universitätsbibliothek): added
  `LANGIDENT_SYSTEMS_LIDS_EXTRA_SUB` and included `SUB/` in
  `LANGIDENT_ENSEMBLE_EXCLUDE_LB_OPTION` (Luxembourgish not predicted).
- **BNF** (Bibliothèque nationale de France): added BNF language support in
  config and ensemble.
- Improved logging of original language metadata for all providers.

## Orchestration

### New Configuration: `v2-0-3`

New recommended config file:
`configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-3.mk`

Key settings:

```makefile
RUN_VERSION_LANGIDENT ?= v2-0-3
CANONICAL_INPUT_KIND  ?= auto
```

This produces run identifiers:

```text
langident-lid-ensemble_multilingual_v2-0-3
langident-lid_stage1-ensemble_multilingual_v2-0-3
```

Default S3 buckets in the config currently point to **staging**:

```makefile
S3_BUCKET_CANONICAL         ?= 111-canonical-staging
S3_BUCKET_LANGIDENT_STAGE1  ?= 114-canonical-processed-staging
S3_BUCKET_LANGIDENT         ?= 114-canonical-processed-staging
```

Override to production buckets in `config.local.mk` for production runs.

### Radio/Audio Targeting

For provider-scoped audio runs, use `CANONICAL_INPUT_KIND=audios`:

```bash
make langident-target \
  CFG=configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-3.mk \
  S3_BUCKET_CANONICAL=111-canonical-staging \
  PROVIDER=RTS \
  NEWSPAPER=ana_media \
  CANONICAL_INPUT_KIND=audios
```

Use `NEWSPAPER_FNMATCH='RTS/*'` for provider-scoped collection runs.

### Other Orchestration Changes

- WIP lock acquisition now supports a `--force` flag for recovery scenarios.
- Removed `--upload-if-newer` option from the default LID configuration.
- Added S3 readiness check scripts under `scripts/`.
- LID system options refactored for clarity: `LANGIDENT_SYSTEMS_LIDS_BASE`,
  `LANGIDENT_SYSTEMS_LIDS_EXTRA_<PROVIDER>`, `LANGIDENT_SYSTEMS_LIDS_EXTRA_DEFAULT`.
- Newspaper job count tuned in config for improved parallel throughput.
- Cookbook submodule updated.

## Dependencies and Models

- `requirements.txt` updated to latest compatible versions.
- `Pipfile` updated; `impresso-pipelines` reference simplified.
- `dotenv.sample` now includes `HF_TOKEN` for the OCRQA pipeline.
- Python 3.11 remains the intended runtime.

## Compatibility and Migration

- `RUN_VERSION_LANGIDENT` changes from `v2-0-2` to `v2-0-3`. Collections
  processed under `v2-0-2` are unaffected; new runs will use `v2-0-3` output
  paths.
- The `alphabetical_ratio` field is now always present in ensemble output JSON.
  Downstream consumers that treated its absence as `None` are unaffected.
- New decision labels (`dominant-by-lowvote`, `dominant-lowvote`) may appear in
  ensemble output for edge cases. Consumers filtering on `lg_decision` should
  handle these values.
- RTS and other radio providers should use
  `CANONICAL_INPUT_KIND=audios` or `CANONICAL_INPUT_KIND=auto`; page-only runs
  will not find audio records.

## Validation

- Python syntax: `python3 -m py_compile lib/impresso_langident_systems.py lib/newspaper_statistics.py lib/impresso_ensemble_lid.py`
- Test suite: `remake test`
- `remake help` verified on macOS.

## Known Issues

- The default `S3_BUCKET_CANONICAL` in `v2-0-3` config points to the staging
  bucket (`111-canonical-staging`). Override in `config.local.mk` for
  production.
- `CANONICAL_INPUT_KIND=auto` relies on Make-level detection via file path
  conventions; confirm correct detection for mixed-media providers before
  large-scale runs.

## Links

- Repository: https://github.com/impresso/impresso-language-identification-cookbook
- Releases: https://github.com/impresso/impresso-language-identification-cookbook/releases
