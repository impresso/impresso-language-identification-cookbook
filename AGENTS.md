## Build and Test Commands

When running local project commands yourself, use `remake` instead of `make`.

Examples:
- Use `remake test`
- Use `remake -n world-test TEST_INPUT_DIR=data/test`
- Use `remake run-baseline RUN_BASELINE_ARGS='--max-docs 1'`

When editing files intended for users, documentation, release notes, README
examples, Makefile help text, or shell snippets, write commands as `make`, not
`remake`.

Do not mention `remake` in public-facing documentation unless explicitly asked.

## Canonical Input Processing

The cookbook supports canonical newspaper pages and radio audio records through
the same language-identification/OCRQA processing path.

- Newspaper canonical input uses `pages/NEWSPAPER-YEAR/*-pages.jsonl.bz2`.
- Radio canonical input uses `audios/NEWSPAPER-YEAR/*-audios.jsonl.bz2`.
- Shared issue metadata uses `issues/NEWSPAPER-YEAR-issues.jsonl.bz2`.
- Use `CANONICAL_INPUT_KIND=auto` by default.
- Use `CANONICAL_INPUT_KIND=pages` or `CANONICAL_INPUT_KIND=audios` only for
  targeted runs and tests.

Do not infer radio/newspaper behavior from provider names. Providers may contain
different media types; rely on the canonical record directory and file naming
conventions.

For provider-scoped collection runs, use `NEWSPAPER_FNMATCH`, for example
`NEWSPAPER_FNMATCH='RTS/*'`.

## Configuration Notes

Use `configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-3.mk`
for canonical langident/OCRQA runs that need both newspaper and radio support.

RTS radio data should not predict Luxembourgish by default. Keep `RTS/` in
`LANGIDENT_ENSEMBLE_EXCLUDE_LB_OPTION` unless explicitly changing that behavior.

When testing staging radio input, override the canonical bucket explicitly:

```bash
make langident-target \
  CFG=configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-3.mk \
  S3_BUCKET_CANONICAL=111-canonical-staging \
  PROVIDER=RTS \
  NEWSPAPER=ana_media \
  CANONICAL_INPUT_KIND=audios
```
