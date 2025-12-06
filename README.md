# Information on impresso language identification (LID)

Identifying the correct language in the multilingual Impresso newspaper
collections is challenging.

Regular LID models are trained on contemporary digital-born texts. OCRized
historical newspapers, however, often contain texts with different spelling
rules and noisy text. Specifically, texts in Gothic fonts that were wrongly
OCRized using Antiqua font settings produce results which can be irritating for
the existing models. Moreover, the identification is of particular difficulty
when dealing with mixed content of Luxemburgish newspapers where a single
article may have several languages. As each content item only features a single
language in our classification schema, the identification results in unsolvable
cases. Other difficulties originate from radio programs, lengthy records of
sports events with many names, which often also confuse standard language
identifier.

The digitized newspapers in our collection differ concerning the available
metadata on the language of their content items:

- missing information (newspapers without any language information)
- partial information (e.g. no information for ads)
- potentially wrong information

As a result, neither the available metadata nor the individual predictions of a
classifier are sufficient to predict the correct language. Therefore, we follow
a three-stage approach:

1. **Stage 1a**: Apply multiple language identification systems (langid, wp_ft,
   impresso_ft, lingua, etc.) to predict the language of each content item
2. **Stage 1b**: Aggregate Stage 1a predictions across all content items to compute
   newspaper-level statistics (dominant language, overall distribution, confidence metrics)
3. **Ensemble Stage**: Make final language decisions using rule-based voting that
   combines Stage 1a predictions with Stage 1b statistics

For our model `impresso_ft`, we selected and trained specifically on items where
the original language was different from the predicted languages, and on
multilingual newspapers from Luxembourg (roughly 2000 content items).

Following these steps, you can produce the language identification JSON files
underlying the Impresso interface and the downstream processing.

## Prerequisites

The build process has been tested on modern Linux and macOS systems and requires
Python 3.11. The project now uses the **Impresso Make-Based Cookbook** for
streamlined processing workflows.

### System Dependencies

Under Debian/Ubuntu, install the following packages:

```sh
# install python3.11 according to your OS
sudo apt install git git-lfs make moreutils parallel python3.11-dev build-essential jq # needed for building
```

On macOS:

```sh
brew install git git-lfs make coreutils parallel jq
```

### Installation

```sh
git clone --recursive https://github.com/impresso/impresso-language-identification-cookbook.git
cd impresso-language-identification-cookbook
python3.11 -mpip install pipenv
PIPENV_VENV_IN_PROJECT=enabled python3.11 -mpipenv install
python3.11 -mpipenv shell
```

### Configuration

Create a `.env` file in the project root with your S3 credentials:

```sh
SE_ACCESS_KEY=your_access_key
SE_SECRET_KEY=your_secret_key
SE_HOST_URL=https://os.zhdk.cloud.switch.ch/
```

Set up the environment:

```sh
$ make setup
$ make create-aws-config
$ make test-aws
```

The cookbook automatically handles data synchronization from S3, eliminating the
need for manual symbolic links or environment variables for data directories.

## Quick Start

For most users, the simplest approach is:

```sh
# Setup (one-time)
make setup

# Process a single newspaper (runs all three stages)
make newspaper NEWSPAPER=actionfem

# Process entire collection in parallel
make collection
```

## Detailed Processing Pipeline

### Stage 1a: Apply Multiple Language Identification Systems

We apply multiple language identification systems to each content item, generating
predictions from each system. Additionally, OCR Quality Assessment (OCRQA) can be
optionally enabled to evaluate the quality of the OCR text for each language.

**Processing Script**: `lib/impresso_langident_systems.py`

**Output**: Per-newspaper-year files with predictions from all configured LID systems

- Example: `WTCH-1828.jsonl.bz2` contains predictions from all systems for year 1828

**Build commands**:

```sh
# Run all three stages in sequence
make langident-target

# Or run individual stages:
make impresso-lid-systems-target     # Stage 1a: Individual system predictions
make impresso-lid-statistics-target  # Stage 1b: Newspaper-level aggregation
make impresso-lid-ensemble-target    # Ensemble: Final decisions
```

### Available Language Identification Systems

The pipeline supports multiple language identification systems that can be configured
via the `LANGIDENT_SYSTEMS_LIDS_OPTION` variable:

```sh
# Use all available systems (default) - recommended for production
make langident-target LANGIDENT_SYSTEMS_LIDS_OPTION="langid impresso_ft wp_ft impresso_langident_pipeline lingua"

# Use only FastText-based systems - faster processing
make langident-target LANGIDENT_SYSTEMS_LIDS_OPTION="impresso_ft wp_ft"

# Include langdetect (note: no Luxembourgish support)
make langident-target LANGIDENT_SYSTEMS_LIDS_OPTION="langid langdetect impresso_ft wp_ft lingua"
```

**Recommendation**: Use the default configuration unless you have specific performance constraints or know that certain languages are not present in your data.

#### Enabling OCR Quality Assessment

To enable OCR quality assessment during language identification:

```sh
# Enable OCRQA with default models
make langident-target LANGIDENT_OCRQA_OPTION="--ocrqa"

# Enable OCRQA with custom Hugging Face repository
make langident-target LANGIDENT_OCRQA_OPTION="--ocrqa" \
    LANGIDENT_OCRQA_REPO_OPTION="impresso-project/OCR-quality-assessment-unigram"

# Enable OCRQA with specific model version
make langident-target LANGIDENT_OCRQA_OPTION="--ocrqa" \
    LANGIDENT_OCRQA_REPO_OPTION="impresso-project/OCR-quality-assessment-unigram" \
    LANGIDENT_OCRQA_VERSION_OPTION="v2.0.0"
```

The OCRQA scores are included in the output JSON and provide quality estimates
for the OCR text in each language. The scores range from 0 to 1, with higher
values indicating better estimated OCR quality.

For processing a single newspaper:

```sh
make newspaper NEWSPAPER=actionfem
```

This step produces a JSON file per year per collection. The cookbook automatically
handles file synchronization and conflict resolution for distributed processing
across multiple machines.

### Properties of Language Identification Tools

The pipeline uses several language identification systems, each with different strengths and designed for different text types:

- **`langid`** - Original langid.py library trained on web texts (supports 97 languages including Luxembourgish):  
  [https://github.com/saffsd/langid.py](https://github.com/saffsd/langid.py)
- **`langdetect`** - Python port of Google's language-detection library (supports 55 languages, but not Luxembourgish):  
  [https://github.com/Mimino666/langdetect](https://github.com/Mimino666/langdetect)  
  **Note: Available but not used by default due to lack of Luxembourgish support**
- **`wp_ft`** - Wikipedia FastText model trained on Wikipedia articles (supports 176 languages including Luxembourgish):  
  [https://fasttext.cc/docs/en/language-identification.html](https://fasttext.cc/docs/en/language-identification.html)
- **`impresso_ft`** - Custom FastText model trained specifically on ~2000 historical newspaper content items from the Impresso collection where original language metadata differed from other LID predictions (recognizes exactly `fr/de/lb/en/it`)
- **`impresso_langident_pipeline`** - Impresso-specific pipeline that combines multiple approaches, from the impresso-pipelines package
- **`lingua`** - Rule-based language detector using n-gram frequency statistics (supports 75 languages including Luxembourgish):  
  [https://github.com/pemistahl/lingua-py](https://github.com/pemistahl/lingua-py)

### Why Multiple Systems?

Historical newspapers present unique challenges that no single language identification system handles perfectly:

- **OCR noise**: Misrecognized characters from historical fonts confuse modern LID systems
- **Mixed content**: Articles may contain foreign names, quotes, or advertisements in different languages
- **Historical spelling**: Older spelling conventions differ from contemporary training data
- **Domain specificity**: News content differs from web texts or Wikipedia articles used to train general LID systems

By combining multiple systems and using ensemble voting, we can leverage the strengths of each approach while mitigating individual weaknesses.

### Stage 1b: Aggregate Newspaper-Level Statistics

Stage 1b aggregates all Stage 1a predictions across a newspaper to compute
newspaper-level statistics. These statistics inform the Ensemble stage's voting
process.

**Processing Script**: `lib/newspaper_statistics.py`

**Input**: All Stage 1a prediction files for a newspaper (e.g., WTCH-1828.jsonl.bz2, WTCH-1829.jsonl.bz2, ...)

**Output**: Single `stats.json` file per newspaper containing:

- Dominant language(s) for the newspaper
- Overall language distribution across all content items
- Confidence in original metadata
- Per-system performance metrics

**Aggregation rules**:

- Content items with less than 200 non-letter characters are ignored.
- Content items with an alphabetical ratio < 0.5 are ignored.
- Every language identification prediction has one vote.
- If external metadata is available (called `orig_lg` henceforth), it also
  counts as a LID prediction.
- If the `impresso_ft` or the `orig_lg` vote has support from at least another
  LID model, their votes are boosted by 1.5 (this boost factor was chosen to give
  additional weight to systems with known reliability on historical content).
- The language with the most votes wins and is counted. In case of a tie, we
  don't count for a specific language.

Whenever the ensemble decision matches the original language information from
the data providers, this counts as positive support. Whenever the original
language information differs from the ensemble decision (excluding any cases
where no decision could be reached), this counts as negative support. The
proportion of positive support assesses the confidence into the original
language information as well as the various LID classifiers. If this threshold
is below 75% we ignore the information when determining the final decision per
content item in stage 2.

To perform this stage, run:

```sh
make impresso-lid-statistics-target
```

**Note**: Stage 1b depends on Stage 1a - all Stage 1a prediction files must exist
before computing newspaper-level statistics.

### Ensemble Stage: Final Language Decisions

The Ensemble stage makes final language decisions by combining:

- **Stage 1a predictions**: Individual system predictions for each content item
- **Stage 1b statistics**: Newspaper-level language distribution and dominant language
- **Original metadata**: When confidence threshold is met
- **OCRQA scores**: When available

**Processing Script**: `lib/impresso_ensemble_lid.py`

**Input**:

- Stage 1a file: `WTCH-1828.jsonl.bz2` (predictions from all systems)
- Stage 1b file: `stats.json` (newspaper-level statistics)

**Output**:

- Final decisions: `WTCH-1828.jsonl.bz2` (final language per content item)
- Diagnostics: `WTCH-1828.diagnostics.json` (decision codes and confidence)

**Decision rules**:

- If the overall support for the original language is below 75%, we ignore it
  completely. Otherwise, the original language is treated the same way as any
  other LID system.

- If all LID systems agree unequivocally, we choose this language. In practice,
  this rule only applies to the languages `de`, `fr`, `en` and `it` due to the
  limitations of the `impresso_ft` system. Decision code: `all`.

- If all LID systems except `impresso_ft` agree on a language other than `de`,
  `fr`, `en` or `it`, and if the language has been selected by the ensemble in
  stage 1b at least once, and if there are at least as many letter characters
  as the minimal text length specifies, accept this other language. This rule
  typically applies for `la`, or other rare languages. Note that while multiple
  systems now support `lb` (Luxembourgish), this rule handles cases where
  `impresso_ft` might disagree due to its specialized training. Decision code: `all-but-impresso_ft`.

- If the text is shorter than 50 characters, we choose the dominant language of
  the newspaper. Decision code: `dominant-by-len`.

- Only if no decision could be made, an ensemble voting is performed. We apply
  a similar voting technique as in the global statistics step of stage 1b in
  which the votes are weighed based on their confidence. The `impresso_ft`
  system receives additional weighting when predicting Luxembourgish (`lb`) due
  to its specialized training on historical newspaper content.

  - If the sum of all votes is below the threshold of `0.5`, we simply choose
    the dominant language of the newspaper. Decision code:
    `dominant-by-lowvote`.
  - Otherwise, the language is set according to the evidence based on weighted
    votes. Decision code: `voting`.

To perform this stage, run:

```sh
make impresso-lid-ensemble-target
```

**Note**:

- The Ensemble stage depends on both Stage 1a (predictions) and Stage 1b (statistics)
- Stage 1b and Ensemble are relatively fast compared to Stage 1a since they process
  existing predictions rather than running LID models on raw text
- Ensemble output includes OCRQA scores (when available) for the decided language

## Output Validation and Diagnostics

The Ensemble stage automatically:

- Validates output against the [Impresso language identification JSON schema](https://github.com/impresso/impresso-schemas/blob/master/json/language_identification/language_identification.schema.json)
- Generates diagnostics files with decision codes and confidence metrics

To aggregate diagnostics across all newspapers:

```sh
make aggregate-langident
```

## Parallelization

The cookbook provides sophisticated parallelization options for efficient processing:

### Parallel Processing Configuration

The build system automatically detects CPU cores and configures optimal parallel processing:

- `NPROC`: Automatically detected number of CPU cores
- `COLLECTION_JOBS`: Number of newspapers to process in parallel (default: 2)
- `NEWSPAPER_JOBS`: Number of parallel jobs per newspaper (auto-calculated)
- `MAX_LOAD`: Maximum system load average for job scheduling

### Performance Tuning Guidelines

- **For CPU-bound tasks**: Set `COLLECTION_JOBS ≤ NPROC`
- **For I/O-bound tasks**: `COLLECTION_JOBS` can exceed `NPROC`
- **High memory usage**: Reduce `COLLECTION_JOBS`
- **System lag**: Reduce `MAX_LOAD` to 70-80% of `NPROC`

### Usage Examples

```sh
# Process full collection with optimal parallelization
make collection

# Process with custom parallel settings
make collection COLLECTION_JOBS=4 MAX_LOAD=8

# Process single newspaper with maximum internal parallelism
make newspaper NEWSPAPER=actionfem NEWSPAPER_JOBS=8

# Override CPU detection for resource limiting
make collection NPROC=16 COLLECTION_JOBS=8
```

### Monitoring Progress

```sh
# Monitor collection processing progress
tail -f build/collection.joblog

# Monitor system resources
htop -u $USER

# Monitor I/O performance
iostat -x 5
```

### Distributed Processing

The cookbook supports distributed processing across multiple machines:

- All data is stored on S3 for shared access
- Local stamp files track progress without conflicts
- Machines can join or leave processing without coordination
- Results are validated and uploaded with integrity checks

To run the full LID process on a single machine with N cores:

```sh
make collection COLLECTION_JOBS=N
```

For distributed processing across multiple machines, simply run the same command
on each machine - the cookbook automatically coordinates work distribution.

## Language Identification from Canonical Pages

When we compute the language identification from canonical pages, we use the same
pipeline as described above, but we set the flag `USE_CANONICAL=1` in
addition to the newspaper name. This flag tells the pipeline to use the canonical
pages as input instead of the rebuilt OCR text. The canonical pages are stored
in a separate S3 bucket and need to be synchronized locally first. The
synchronization is done automatically by the cookbook when the flag
`USE_CANONICAL=1` is set.
The canonical pages are stored in the S3 bucket defined by the variable
`S3_BUCKET_CANONICAL`. By default, this variable is set to `112-canonical-sandbox`
in the configuration file `configs/config-langidentocrqa_canonical-lid-ensemble_multilingual_v2-0-1.mk`.
You can override this variable in your own configuration file.
Assuming that $(NEWSPAPER) contains the provider level prefix, and $(NP) is the newspaper name without provider prefix,
The path arithmetic for canonical processing is as follows:

- The canonical pages of a NEWSPAPER for each YEAR are stored in the S3 bucket $(S3_BUCKET_CANONICAL)
  under the path
  `$(NEWSPAPER)/pages/$(NP)-$(YEAR)/$(NP)-YEAR-MM-DD-EDITION-pages.jsonl.bz2`.
  - The local stamps for the make build refer
    $(BUILD_DIR)/$(NEWSPAPER)/pages/$(NP)-YEAR.stamp reflect the newest
    $(NP)-YEAR-MM-DD-EDITION-pages.jsonl.bz2 timestamp metadata on S3.
  - The processed canonical LID output files are stored per NEWSPAPER and YEAR packages in the
    bucket $(S3_BUCKET_CANONICAL_PROCESSED_DATA) under the path
    `$(NEWSPAPER)/langident/RUNID/systems/$(NP)-YEAR-lid.jsonl.bz2`.
  - In order to compute
