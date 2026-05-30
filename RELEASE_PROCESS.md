# Release Process Guide

This document describes how to prepare and publish releases for the Impresso
language identification cookbook repository.

This repository is a processing pipeline, not a library-only project. A release
is a tagged repository snapshot that captures changes to:

- language identification logic in `lib/impresso_langident_systems.py`,
  `lib/newspaper_statistics.py`, and `lib/impresso_ensemble_lid.py`,
- Make orchestration in `Makefile`, `cookbook/*.mk`, and `configs/*.mk`,
- dependency definitions in `Pipfile`, `Pipfile.lock`, and `requirements.txt`,
- LID models and resources under `models/`,
- run configuration files in `configs/`,
- operational documentation such as `README.md`, `AGENTS.md`, and this file.

The release must make clear whether it changes the repository code, the
language identification output contract, or the default `RUN_ID_LANGIDENT`.

## Table of Contents

- [Release Workflow](#release-workflow)
- [Version Naming](#version-naming)
- [Preparing a Release](#preparing-a-release)
- [Release Notes](#release-notes)
- [Publishing a Release](#publishing-a-release)
- [Post-Release Tasks](#post-release-tasks)
- [Hotfix Releases](#hotfix-releases)
- [Checklist](#checklist)

## Release Workflow

### Overview

Releases follow these steps:

1. Prepare the repository state for release.
2. Review pipeline, dependency, cookbook, and configuration changes.
3. Update release documentation.
4. Write and commit release notes before tagging.
5. Open and merge a pull request into `master`.
6. Create an annotated git tag from the merged commit on `master`.
7. Publish a GitHub release from the committed release notes file.
8. Perform post-release verification.

The main rule is that release notes must be committed before the tag is created.
The tag, release notes, README release entry, and published GitHub release should
all refer to the same repository snapshot.

The normal path is therefore:

1. prepare release changes on a branch,
2. merge that branch through a pull request,
3. switch to the updated `master`,
4. tag the merged commit on `master`,
5. publish the GitHub release from that tag.

Creating a release directly from a feature branch should be treated as an
exception.

## Version Naming

This repository uses a single version concept: a **semantic git tag** that
identifies both the repository snapshot and the pipeline run version embedded
in S3 output paths.

### Git tags and pipeline run versions — semantic scheme

Git tags use semantic versioning:

```text
vMAJOR.MINOR.PATCH
```

Examples:

- `v2.0.2`
- `v2.1.0`

Historical tags include older pipeline-like names and semantic versions:

- `v1.2`, `v1.4`, `v1.4.1` … `v1.4.4` — first-generation releases
- `v2.0.1`, `v2.0.2` — current generation

The tag is directly tied to the **pipeline run version**, a semantic identifier
carried by the Make variable `RUN_VERSION_LANGIDENT` and embedded in
`RUN_ID_LANGIDENT`:

```makefile
RUN_VERSION_LANGIDENT ?= v2-0-2
```

This produces run identifiers such as:

```text
langident-lid-ensemble_multilingual_v2-0-2
langident-lid_stage1-ensemble_multilingual_v2-0-2
```

The version scheme mirrors the S3 data:

- **Major** (`vX.0.0`): incompatible output format change, new schema, or
  processing semantics that require a full rerun of all newspapers.
- **Minor** (`vX.Y.0`): new supported language, new LID system, new canonical
  input kind, or additive output change that does not break existing consumers.
- **Patch** (`vX.Y.Z`): bug fix, configuration correction, or operational change
  that does not affect output content for already-processed data.

Update `RUN_VERSION_LANGIDENT` intentionally whenever a release changes LID
output schema, default model set, or processing semantics, and document the
resulting `RUN_ID_LANGIDENT` in the release notes.

Use the same pipeline run version consistently in:

- `cookbook/paths_langident.mk` or selected `configs/*.mk`,
- `README.md` release notes,
- release notes operational guidance,
- any S3 output path examples affected by the change.

Use the same git tag string consistently in:

- the git tag,
- the release notes filename (`RELEASE_NOTES_<tag>.md`),
- the GitHub release title.

## Preparing a Release

### 1. Review the Changes

Inspect all commits and changed files since the previous release tag:

```bash
git log <previous-tag>..HEAD --oneline
git diff <previous-tag>..HEAD --stat
git diff <previous-tag>..HEAD --name-status
```

Focus especially on these areas:

- `Makefile`
- `README.md`
- `AGENTS.md`
- `RELEASE_PROCESS.md`
- `Pipfile`
- `Pipfile.lock`
- `requirements.txt`
- `config.local.mk.sample`
- `configs/`
- `lib/`
- `models/`
- `cookbook/`

Useful targeted review commands:

```bash
git log <previous-tag>..HEAD --oneline -- lib/
git log <previous-tag>..HEAD --oneline -- cookbook/
git diff <previous-tag>..HEAD -- cookbook/paths_langident.mk configs/
git diff <previous-tag>..HEAD -- Pipfile Pipfile.lock requirements.txt
```

### 2. Review Language Identification Changes

For changes to `lib/impresso_langident_systems.py`, `lib/newspaper_statistics.py`,
or `lib/impresso_ensemble_lid.py`, confirm whether the release changes any output
contract or processing semantics:

- supported LID systems (`LANGIDENT_SYSTEMS_LIDS_BASE`,
  `LANGIDENT_SYSTEMS_LIDS_EXTRA_*`),
- Stage 1a predictions: new or removed classifier outputs,
- Stage 1b statistics: newspaper-level aggregation logic,
- Ensemble stage: voting rules or confidence thresholds,
- Luxembourgish exclusion logic (`LANGIDENT_ENSEMBLE_EXCLUDE_LB_OPTION`),
- canonical input kind support (`CANONICAL_INPUT_KIND`: pages, audios, auto),
- JSON schema validation behavior,
- output field names or confidence score formats.

If any of these change, document the impact in release notes and decide whether
`RUN_VERSION_LANGIDENT` must change.

### 3. Review Make and S3 Orchestration Changes

Review changes to path, sync, processing, and upload behavior:

- S3 bucket defaults such as `S3_BUCKET_CANONICAL`, `S3_BUCKET_LANGIDENT_STAGE1`,
  and `S3_BUCKET_LANGIDENT`,
- `RUN_ID_LANGIDENT` and `RUN_ID_LANGIDENT_STAGE1` construction,
- canonical input path handling and `CANONICAL_INPUT_KIND` behavior,
- WIP marker behavior and max age settings,
- `local_to_s3` upload options,
- `NEWSPAPER`, `NEWSPAPER_FNMATCH`, provider, and collection-list handling,
- `COLLECTION_JOBS`, `NEWSPAPER_JOBS`, and `MAX_LOAD`.

Do not change S3 path conventions or stamp-file semantics without calling that
out explicitly as a breaking operational change.

### 4. Review Dependency and Model State

Check:

- `Pipfile`
- `Pipfile.lock`
- `requirements.txt`
- `pyproject.toml`
- `models/`

Questions to answer before release:

- Did dependency changes come from an intentional lock refresh?
- Is Python 3.11 still the intended runtime?
- Did bundled FastText model metadata or licensing notes change?
- Are LID model versions documented?

### 5. Update Documentation

Review and update documentation when the release changes behavior, supported
languages, targets, dependencies, or operational workflow.

Common files to review:

- `README.md`
- `AGENTS.md`
- `config.local.mk.sample`
- `dotenv.sample`
- `cookbook/README.md` if shared cookbook behavior changed

This repository currently keeps historical release notes in `README.md` rather
than a root-level `CHANGELOG.md`. For a release, either update the README
release notes section or add a root changelog deliberately and use it
consistently going forward. In either case, keep release entries in newest-first
order, with the newest release at the top.

### 6. Perform Release Verification

Use lightweight checks first:

```bash
python3 -m py_compile lib/impresso_langident_systems.py lib/newspaper_statistics.py lib/impresso_ensemble_lid.py
make help
```

On macOS, run Make commands through `remake` or `gmake`, while keeping examples
written as `make` in documentation.

If dependency setup changed, also check the project environment:

```bash
pipenv run python --version
pipenv run python -m py_compile lib/impresso_langident_systems.py lib/newspaper_statistics.py lib/impresso_ensemble_lid.py
```

Run the test suite with a small sample:

```bash
remake test
```

If you have valid S3 credentials and intentionally want runtime verification,
run a small known-good target. This is optional and environment-dependent:

```bash
make langident-target NEWSPAPER=<provider>/<newspaper> NEWSPAPER_JOBS=1 MAX_LOAD=1
```

### 7. Prepare the Pull Request

Before releasing, ensure the release-ready branch is reviewed and merged into
`master`.

Recommended checks before opening the PR:

- the branch contains release notes and README/changelog updates,
- the working tree is clean,
- the branch is pushed to GitHub,
- the branch diff against `master` matches the intended release scope.

Useful commands:

```bash
git status --short
git log --oneline origin/master..HEAD
git diff --stat origin/master..HEAD
git push origin <release-branch>
```

Then open the PR:

```bash
gh pr create --base master --head <release-branch>
```

Or open it through the GitHub web interface.

After approval, resolve and merge the PR manually in GitHub. Do not use
automatic PR merge commands from the CLI for the normal release flow.

### 8. Sync Local `master` Before Tagging

Do not create the release tag from the feature branch. After the PR is merged:

```bash
git checkout master
git pull --ff-only origin master
git log -1 --oneline
```

Verify that the top commit on `master` is the merged release commit that contains:

- `README.md` release notes or root `CHANGELOG.md`,
- `RELEASE_NOTES_<tag>.md`,
- all intended pipeline, configuration, dependency, and documentation changes.

## Release Notes

Release notes should be created before tagging and committed together with the
final release-ready state. Historical release-note or changelog sections should
be maintained in newest-first order.

### Filename

Use the exact git tag string in the filename for the committed release-notes file:

```text
RELEASE_NOTES_<tag>.md
```

Examples:

- `RELEASE_NOTES_v2.1.0.md`
- `RELEASE_NOTES_v2.0.3.md`

### Suggested Structure

Release notes for this repository should focus on processing impact,
reproducibility, S3 output paths, and operational changes.

Suggested sections:

1. Overview
2. Language identification changes
3. Make/S3 orchestration changes
4. Dependency and model changes
5. Output compatibility and migration notes
6. Validation performed
7. Known limitations or follow-up work

Template:

```markdown
# Release Notes - <tag>

**Release Date:** YYYY-MM-DD
**Tag:** <tag>
**Status:** Stable

## Overview

Brief summary of the release.

## Language Identification

- Changes to LID systems, supported languages, schemas, or output fields.
- Whether `RUN_VERSION_LANGIDENT` changed.
- Resulting `RUN_ID_LANGIDENT` and `RUN_ID_LANGIDENT_STAGE1`, if relevant.

## Orchestration

- Changes to Make targets, S3 paths, canonical input handling, WIP behavior,
  stamps, or parallelism.

## Dependencies and Models

- Python, FastText, LID model, or cookbook dependency changes.

## Compatibility and Migration

- Any rerun, recomputation, S3 cleanup, or downstream migration required.

## Validation

- Checks performed before release.

## Known Issues

- Known limitations, if any.

## Links

- Repository: https://github.com/impresso/impresso-language-identification-cookbook
```

### Generating Change Lists

Use git to build the release notes content:

```bash
git log <previous-tag>..HEAD --oneline
git shortlog <previous-tag>..HEAD -sn
git diff <previous-tag>..HEAD --stat
git diff <previous-tag>..HEAD --name-status
git diff <previous-tag>..HEAD -- lib/ cookbook/
git diff <previous-tag>..HEAD -- Makefile README.md AGENTS.md
git diff <previous-tag>..HEAD -- Pipfile Pipfile.lock requirements.txt
```

## Publishing a Release

### 1. Commit Release Notes and Final Metadata

Before tagging, commit the release notes and any final release-ready updates:

```bash
git add README.md AGENTS.md RELEASE_PROCESS.md RELEASE_NOTES_<tag>.md
git commit -m "Prepare release <tag>"
```

Adjust the staged file list to match the actual release contents. Depending on
the release, you may also include:

- `Makefile`
- `Pipfile`
- `Pipfile.lock`
- `requirements.txt`
- `config.local.mk.sample`
- `configs/*`
- `lib/*`
- `cookbook/*`
- `models/*`

### 2. Open and Merge the Pull Request

Push the release branch and merge it into `master` before creating the tag.

The pull request should be resolved manually in GitHub.

Example:

```bash
git push origin <release-branch>
gh pr create --base master --head <release-branch>
```

Then review and merge the PR manually in the GitHub web UI, and afterwards sync
local `master`:

```bash
git checkout master
git pull --ff-only origin master
```

### 3. Create the Git Tag

Create an annotated tag from the exact merged release commit on `master`:

```bash
git tag -a <tag> -m "Release <tag>"
git push origin <tag>
```

Example:

```bash
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
```

Before tagging, confirm you are on `master` and on the merged release commit:

```bash
git branch --show-current
git log -1 --oneline
```

### 4. Create the GitHub Release

#### Via GitHub Web Interface

1. Go to https://github.com/impresso/impresso-language-identification-cookbook/releases
2. Click "Draft a new release"
3. Select the tag you just pushed
4. Use the tag string as the title, or add a short descriptive suffix
5. Paste the contents of the committed `RELEASE_NOTES_<tag>.md` file into the description
6. Publish the release

#### Via GitHub CLI

```bash
gh auth login

gh release create <tag> \
   --title "<tag>" \
   --notes-file RELEASE_NOTES_<tag>.md
```

Example:

```bash
gh release create v2.1.0 \
   --title "v2.1.0" \
   --notes-file RELEASE_NOTES_v2.1.0.md
```

Using `--notes-file` is preferred because the published GitHub release text then matches the committed release notes file in git.

### 5. Correcting an Existing Release

If you must fix a published release description after the fact:

```bash
gh release edit <tag> --notes-file RELEASE_NOTES_<tag>.md
```

Treat this as an exception path. The normal flow is to finalize the release
notes before tagging.

## Post-Release Tasks

### 1. Verify the Published Release

After publishing:

- confirm the tag exists on GitHub,
- confirm the GitHub release points to the correct commit,
- confirm the tagged commit is reachable from `master`,
- confirm the release notes match the committed file,
- confirm the README release entry or changelog entry is present in the tagged
  snapshot.

### 2. Verify Clone and Setup Instructions

Ensure the documented setup flow still makes sense for the new release:

```bash
git clone --recursive https://github.com/impresso/impresso-language-identification-cookbook.git
cd impresso-language-identification-cookbook
python3.11 -mpip install pipenv
python3.11 -mpipenv install
python3.11 -mpipenv shell
make help
```

On macOS, use `remake help` or `gmake help`.

If the release changed environment assumptions, update:

- `README.md`
- `config.local.mk.sample`
- `dotenv.sample`

### 3. Notify Stakeholders

Depending on the release, notify relevant team members about:

- new `RUN_ID_LANGIDENT` or `RUN_ID_LANGIDENT_STAGE1` values,
- changed supported languages or LID systems,
- output format or schema changes,
- required reruns or recomputation,
- S3 bucket/path changes,
- dependency or environment changes.

### 4. Monitor for Follow-up Issues

After release:

- watch GitHub issues,
- track failures in S3 processing runs,
- inspect logs for WIP lock or upload failures,
- be prepared to issue a follow-up tag if a run ID, dependency, model, or S3 path
  was wrong.

### 5. If a Release Was Created from the Wrong Branch

If a release was published from a feature branch before the PR was merged:

1. delete the GitHub release,
2. delete the incorrect tag locally and on origin if appropriate,
3. merge the branch through a PR into `master`,
4. recreate the tag from `master`,
5. republish the GitHub release.

Example cleanup commands:

```bash
git tag -d <tag>
git push origin :refs/tags/<tag>
```

Only do this if you are intentionally replacing the release and have confirmed
the team agrees with rewriting that published tag.

## Hotfix Releases

Use a hotfix release when a published tag contains an incorrect configuration,
broken target wiring, or other release-critical issue.

Typical cases:

- wrong `RUN_VERSION_LANGIDENT` or `RUN_ID_LANGIDENT`,
- wrong S3 bucket or prefix,
- broken Make target,
- broken LID model or FastText model dependency,
- incorrect `Pipfile.lock`,
- missing documentation for required reruns or output migration.

Suggested process:

```bash
git checkout -b hotfix/<tag-fix> <released-tag>
# make the fix
git commit -am "Fix release issue for <new-tag>"
git tag -a <new-tag> -m "Hotfix release <new-tag>"
git push origin <new-tag>
```

Then create a focused GitHub release with notes describing exactly what was
corrected.

## Checklist

- [ ] Reviewed commits and file changes since the previous tag
- [ ] Reviewed LID changes and output compatibility
- [ ] Reviewed Make/S3 orchestration, WIP, and stamp behavior
- [ ] Reviewed `Pipfile`, `Pipfile.lock`, and `requirements.txt`
- [ ] Reviewed bundled model assets and LID model versions
- [ ] Updated `README.md` release notes or root changelog
- [ ] Updated `AGENTS.md` or config samples if needed
- [ ] Decided whether `RUN_VERSION_LANGIDENT` must change
- [ ] Verified Python syntax
- [ ] Verified `make help` with GNU Make, `remake`, or `gmake`
- [ ] Ran `remake test` to verify basic functionality
- [ ] Wrote `RELEASE_NOTES_<tag>.md` before tagging
- [ ] Committed release notes on the release branch
- [ ] Opened PR and resolved it manually in GitHub
- [ ] Merged PR into `master`
- [ ] Synced local `master` to the merged release commit
- [ ] Created and pushed annotated git tag from `master`
- [ ] Published GitHub release from the committed release notes file
- [ ] Performed post-release verification

## Tools and Resources

- GitHub repository: https://github.com/impresso/impresso-language-identification-cookbook
- GitHub releases: https://github.com/impresso/impresso-language-identification-cookbook/releases
- GitHub issues: https://github.com/impresso/impresso-language-identification-cookbook/issues
- GitHub CLI: https://cli.github.com/
- Keep a Changelog: https://keepachangelog.com/
- Git tagging documentation: https://git-scm.com/book/en/v2/Git-Basics-Tagging

## Questions?

If you have questions about the release process:

- review recent tags and release notes,
- check previous GitHub releases for examples,
- inspect `README.md` release notes for historical pipeline changes,
- ask maintainers before publishing a tag that changes `RUN_ID_LANGIDENT`,
  output formats, or S3 path conventions.

---

**Last Updated:** 2026-05-26
