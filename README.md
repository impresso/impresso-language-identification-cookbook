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
a three-step approach:

1. predict the language of an article using various probabilistic language
   identification classifiers (stage 1a)
2. aggregate the predictions, compute an ensemble decision for longer articles
   and assess the confidence of a classifier by comparing against the ensemble
   decision (stage 1b)
3. predict the final language of an article following a rule-based approach and
   ensemble voting (ensemble stage)

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
python3.11 -mpipenv install
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

# Process a single newspaper
make newspaper NEWSPAPER=actionfem

# Process entire collection (parallel)
make collection
```

## Detailed Processing Pipeline

### Stage 1a: Automatic Language Identification

We first apply several off-the-shelf LID classifiers and our model to the
texts. Additionally, OCR Quality Assessment (OCRQA) can be optionally enabled
to evaluate the quality of the OCR text for each language. The OCRQA scores
provide an estimate of text quality that can be used alongside language
identification for assessing the reliability of the results.

The corresponding build command is:

```sh
make langident-target
```

This command runs all three stages (1a, 1b, and ensemble) in sequence. To run individual stages:

```sh
make impresso-lid-stage1a-target  # Initial LID predictions only
make impresso-lid-stage1b-target  # Collection statistics only
make impresso-lid-ensemble-target # Final ensemble decisions only
```

# To use the new script name in any documentation that mentions the specific script:

```sh
# Stage 1a uses impresso_langident_systems.py
# Stage 1b uses newspaper_statistics.py
# Ensemble uses impresso_ensemble_lid.py
```

### Available Language Identification Systems

The pipeline supports multiple language identification systems that can be configured
via the `LANGIDENT_LID_SYSTEMS_OPTION` variable:

```sh
# Use all available systems (default) - recommended for production
make langident-target LANGIDENT_LID_SYSTEMS_OPTION="langid impresso_ft wp_ft impresso_langident_pipeline lingua"

# Use only FastText-based systems - faster processing, good for major languages
make langident-target LANGIDENT_LID_SYSTEMS_OPTION="impresso_ft wp_ft"

# Include langdetect for additional coverage - use when Luxembourgish is not expected
make langident-target LANGIDENT_LID_SYSTEMS_OPTION="langid langdetect impresso_ft wp_ft lingua"
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

### Stage 1b: Aggregating collection statistics on language

Given the incomplete and sometimes unreliable metadata regarding the content
items' language, we aggregate statistics per collection to assess the confidence
in the classifiers. The global statistics allow us to take a more informed
decision in the next stage of processing.

In order to assess the dominant language of a newspaper, we compute the
statistics per collection according to the following rules:

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

To perform this stage, run the following command:

```sh
make impresso-lid-stage1b-target
```

This command can only be run after stage 1a has been completed. It processes
the aggregated statistics for the entire collection and must be run sequentially.

### Stage Ensemble: Deciding the language per content item

Given the output from various LID systems, the original language information,
and optional OCRQA scores, we finally decide the language of an article
according to the following rules:

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

To perform this stage, run the following command:

```sh
make impresso-lid-ensemble-target
```

The process of stage 1b and ensemble is relatively fast compared to stage 1a since it processes the already-computed predictions rather than running the language identification models on raw text.

**Note**: The ensemble output includes OCRQA scores (when available from stage 1a)
in the final JSON. For each content item, the output contains the OCRQA score and
model identifier for the decided language, which can be used to assess the
reliability of the OCR text quality.

## Preparing the data release

Preparing the LID data release involves the following steps:

- Validating the jsonl files from stage 2 according to impresso's [language
  identification JSON
  schema](https://github.com/impresso/impresso-schemas/blob/master/json/language_identification/language_identification.schema.json).
- Copying over the per-collection aggregation statistics from stage 1b.
- Preparing statistical diagnostics files for the whole impresso collection set

```sh
make impresso-lid-release-target
```

After preparing the data one can upload to a configured s3 bucket

```sh
make impresso-lid-upload-release-to-s3
```

## Creating LID statistics

During the ensemble stage, diagnostics files in JSON format are produced for each collection
that aggregate information from the individual content item files. These statistics
can be aggregated across the entire collection:

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
