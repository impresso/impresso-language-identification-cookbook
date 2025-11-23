#!/usr/bin/env python3
"""
Aggregate language-related statistics on content items to assess
the overall confidence into different classifiers for language identification (LID).

This module implements Stage 1b of the impresso language identification pipeline:
aggregating newspaper statistics to assess classifier confidence and determine
dominant languages per newspaper.

Given the incomplete and sometimes unreliable metadata regarding content items'
language, this module aggregates statistics per newspaper to assess confidence
in the classifiers. The global statistics allow for more informed decisions
in subsequent processing stages.

Key features:
- Ensemble voting with configurable boost factors for specific LID systems
- Statistical assessment of original language metadata reliability
- Filtering based on text length and alphabetical ratio thresholds
- Support for multiple LID systems: langdetect, langid, impresso_ft, wp_ft,
  impresso_langident_pipeline, lingua
- S3 and local file support for input
- Comprehensive logging and statistical reporting

Type Safety:
- Provides typed aliases for content items to facilitate static analysis across downstream tools.

Ensemble Decision Rules:
- Content items with less than 200 non-letter characters are ignored by default
- Content items with alphabetical ratio < 0.5 are ignored
- Every language identification prediction has one vote
- If external metadata (orig_lg) is available, it counts as a LID prediction
- If impresso_ft or orig_lg votes have support from at least another LID model,
  their votes are boosted by a configurable factor (default 1.5)
- The language with the most votes wins; ties result in no decision

Support Assessment:
- When ensemble decision matches original language information: positive support
- When original language differs from ensemble decision: negative support
- Support ratio below 75% indicates unreliable original language metadata

Input Format:
JSON Lines format with LID predictions per content item:

{
   "tp":"page",
   "id":"arbeitgeber-1909-01-02-a-i0017",
   "len":5636,
   "orig_lg":null,
   "alphabetical_ratio":0.79,
   "langdetect": [{"lang": "de", "prob": 1.0}],
   "langid": [{"lang": "de", "prob": 1.0}],
   "impresso_ft": [{"lang": "de", "prob": 1.0}],
   "wp_ft": [{"lang": "de", "prob": 0.95}, {"lang": "en", "prob": 0.01}]
}

Output Format:
JSON object with newspaper-level statistics including:
- Language frequency distributions per LID system
- Support ratios for each LID system and original metadata
- Dominant language and overall confidence metrics
- Content type and length distributions

Example usage:
    aggregator = AggregatorLID(
        infile=["stage1a_output.jsonl"],
        newspaper="newspaper_newspaper",
        lids={"langdetect", "langid", "impresso_ft", "wp_ft"},
        boosted_lids={"impresso_ft", "orig_lg"},
        boost_factor=1.5,
        minimal_vote_score=1.5,
        minimal_lid_probability=0.25,
        minimal_text_length=200,
        round_ndigits=9,
        admissible_languages=None,
        git_describe=""
    )
    aggregator.run()
"""

__version__ = "2025.10.10"

import json
import logging
import time
import re
import sys

from collections import Counter, defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
    Generator,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TextIO,
)

import smart_open


from impresso_cookbook import (
    get_s3_client,
    get_timestamp,
    setup_logging,
    yield_s3_objects,
)

log = logging.getLogger(__name__)

ContentItem = Dict[str, Any]
Language = str
LIDName = str
VoteScore = float
VoteRecord = Tuple[LIDName, VoteScore]
VotesByLanguage = DefaultDict[Language, List[VoteRecord]]
VoteTally = Dict[Language, VoteScore]
FrequencyMapping = DefaultDict[Language, float]


def update_relfreq(
    counter: MutableMapping[Language, float],
    n: Optional[int] = None,
    ndigits: int = 3,
) -> None:
    """Normalize a language frequency distribution in-place.

    Args:
        counter: Mutable frequency distribution keyed by language.
        n: Optional total count to reuse instead of recomputing.
        ndigits: Number of decimal places used for rounding ratios.
    """
    total = float(sum(counter.values())) if n is None else float(n)
    if total == 0.0:
        return
    for lang in list(counter.keys()):
        counter[lang] = round(float(counter[lang]) / total, ndigits)


def expand_s3_prefix(s3_path: str) -> List[str]:
    """Expand an S3 prefix into all matching JSONL resources.

    Args:
        s3_path: Fully qualified S3 URI (``s3://bucket/prefix``).

    Returns:
        List[str]: JSONL object keys under the provided prefix.
    """
    match = re.match(r"s3://([^/]+)/(.+)", s3_path)
    if not match:
        raise ValueError(f"Invalid S3 path format: {s3_path}")

    bucket, prefix = match.groups()

    # Get all objects with the prefix and filter for jsonl.bz2 files
    files = []
    for obj_key in yield_s3_objects(bucket, prefix):
        if obj_key.endswith(".jsonl.bz2"):
            files.append(f"s3://{bucket}/{obj_key}")

    log.info("Found %d matching files for prefix %s", len(files), s3_path)
    return files


def extract_prediction(predictions: Any) -> Optional[Tuple[Language, Optional[float]]]:
    """Return the leading (language, probability) pair from a prediction list."""
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            lang_value = first.get("lang")
            prob_value = first.get("prob")
            if isinstance(lang_value, str):
                prob_float: Optional[float] = None
                if isinstance(prob_value, (int, float)):
                    prob_float = float(prob_value)
                return lang_value, prob_float
    return None


class AggregatorLID:
    """Assess confidence of multiple language identifiers based on global statistics.

    :param str infile: JSON file containing the language predictions per content item.
    :param str newspaper: Short canonical name of newspaper.
    :param Set[str] lids: Set of LID systems predict language/probability pairs.
        Therefore, orig_lg is not seen as LID system as it "predicts" only a single language if any.
    :param Set[str] boosted_lids: Set of LIDs that are boosted by a boost factor.
    :param float boost_factor: Boost factor applied to boosted LIDS if they have
        support from at least another LID. The idea is that on their own some of
        the LIDs or the `orig_lg` can be pretty wrong. If they have at least a
        support from another system the confidence into their decision grows considerably.
    :param Optional[Set[str]] admissible_languages: Limit languages in the ensemble decisions.
        If None, no restrictions are applied.
    :param float minimal_vote_score: Minimal vote score from ensemble to reach a decision.
    :param float minimal_lid_probability: Minimal probability from a LID decision to be considered a vote.
    :param int minimal_text_length: Threshold on article length in chars for computing LID support by ensemble.
    :param str git_describe: Output of git describe to use as version if not empty string
    :param int round_ndigits: Number of decimal places in the output.

    :attr str version: Version of the newspaper script.
    :attr list attrs_for_json: Defines all attributes of this data object that
        enter the JSON output in their corresponding order.
    :attr Optional[float] total_orig_support_ratio: Percentage of all content items
        with a non-null original language and a minimal length threshold
        where the original language matches the ensemble decision.
    :attr Optional[float] overall_orig_lg_support: Percentage of existing language
        categorizations (i.e. `orig_lg`) that is backed by the ensemble decision.
        This number serves as an overall criterion on the confidence that we can establish for a newspaper.
    :attr int n: Total number of content items that are not filtered out due to
        incompatible type (img) or lack of any textual content.
    :attr str dominant_language: The most frequent language of a newspaper according to the ensemble decision.
        The detailed percentage for this language can be found in the language
        class distribution in the ensemble frequency distribution.
        This value is extracted for convenience here.
    :attr dict lg_support: Counter about agreement/disagreement w.r.t.
        the ensemble decision for each selected LID and `orig_lg`.
    :attr dict lid_distributions: Counter with a language frequency distribution
        for each selected LID, `orig_lg` and the voting results `ensemble`.
    :attr Counter contentitem_type_distribution: Distribution of content item types (article, ad, image etc.).
    :attr Counter content_length_stats: Distribution of article lengths (raw character counts).
    :attr Counter orig_lg_ensemble_disagreements: Count of disagreements between orig_lg and ensemble decisions,
        formatted as "orig_lang->ensemble_lang".
    :attr int orig_lg_total_decisions: Total number of content items with non-null orig_lg that were processed.

    """

    def __init__(
        self,
        infile: List[str],
        newspaper: Optional[str],
        lids: Set[str],
        boosted_lids: Set[str],
        boost_factor: float,
        minimal_vote_score: float,
        minimal_lid_probability: float,
        minimal_text_length: int,
        round_ndigits: int,
        admissible_languages: Optional[Set[str]],
        git_describe: str,
        outfile: Optional[str] = None,
    ):
        self.attrs_for_json: List[str] = [
            # configured information
            "newspaper",
            "lids",
            "boosted_lids",
            "boost_factor",
            "admissible_languages",
            # collected statistical information
            "dominant_language",
            "overall_orig_lg_support",
            "n",
            "lid_distributions",
            "lid_absolute_counts",
            "lg_support",
            "contentitem_type_distribution",
            "orig_lg_ensemble_disagreements",
            "orig_lg_total_decisions",
            # administrative information
            "ts",  # Add timestamp at top level
            "aggregator_lid",
        ]
        self.output: Optional[TextIO] = (
            smart_open.open(outfile, mode="w", encoding="utf-8") if outfile else None
        )
        self.start_time: Optional[float] = None
        self.s3_client = get_s3_client()
        self.ts = get_timestamp()

        self.aggregator_lid: dict = {
            "ts": self.ts,
            "version": git_describe or __version__,
        }

        # Expand S3 prefixes to actual file lists
        expanded_files = []
        for input_path in infile:
            if input_path.startswith("s3://") and not input_path.endswith(".jsonl.bz2"):
                # This looks like an S3 prefix, expand it
                log.info("Expanding S3 prefix: %s", input_path)
                expanded_files.extend(expand_s3_prefix(input_path))
            else:
                # Regular file path or complete S3 path
                expanded_files.append(input_path)

        self.infile: List[str] = expanded_files
        log.info("Processing %d input files", len(self.infile))

        self.newspaper: Optional[str] = newspaper

        self.lids: Set[str] = set(lid for lid in lids if lid != "orig_lg")

        if len(self.lids) < 1:
            log.error(
                "No LID models provided. At least one language identificator needed."
            )
            exit(2)

        self.total_orig_support_ratio: Optional[float] = None

        self.boosted_lids: Set[str] = set(
            lid for lid in boosted_lids if lid == "orig_lg" or lid in self.lids
        )

        if self.boosted_lids != set(boosted_lids):
            log.warning(
                "The set of boosted_lids contained the following invalid and ignored"
                " system identifiers:"
                f" {self.boosted_lids.symmetric_difference(boosted_lids)}"
            )

        self.boost_factor: float = boost_factor

        self.minimal_vote_score: float = minimal_vote_score

        self.minimal_lid_probability: float = minimal_lid_probability

        self.minimal_text_length: int = minimal_text_length

        self.round_ndigits: int = round_ndigits

        self.admissible_languages: Optional[Set[str]] = (
            set(admissible_languages) if admissible_languages else None
        )

        self.overall_orig_lg_support: Optional[float] = None

        self.n: int = 0

        self.dominant_language: Optional[str] = None

        self.lg_support: Dict[str, DefaultDict[Language, float]] = {
            lid: defaultdict(float) for lid in self.lids.union(("orig_lg",))
        }

        self.lid_distributions: Dict[str, FrequencyMapping] = {
            lid: defaultdict(float) for lid in self.lids.union(("orig_lg", "ensemble"))
        }

        # Absolute counts for each LID system before conversion to relative frequencies
        self.lid_absolute_counts: dict = {
            lid: Counter() for lid in self.lids.union(("orig_lg", "ensemble"))
        }

        self.contentitem_type_distribution: Counter = Counter()

        self.content_length_stats: Counter = Counter()

        # Statistics for orig_lg vs ensemble disagreements
        self.orig_lg_ensemble_disagreements: Counter = Counter()
        self.orig_lg_total_decisions: int = 0

    def run(self) -> None:
        """Execute the aggregation workflow and emit the final JSON payload."""
        self.start_time = time.time()

        log.info(
            "Starting language statistics aggregation for input files: %s",
            ", ".join(self.infile),
        )
        log.info("Using LID systems: %s", ", ".join(self.lids))

        self.collect_statistics()
        self.compute_support()
        json_data = self.jsonify()

        print(
            json.dumps(json_data, ensure_ascii=False),
            file=self.output if self.output is not None else sys.stdout,
        )

        if self.start_time is None:
            raise RuntimeError("Aggregation timer was not initialized.")
        total_time = time.time() - self.start_time
        log.info(
            "Language statistics aggregation finished in %.2f seconds.", total_time
        )

    def get_next_contentitem(self) -> Generator[ContentItem, None, None]:
        """Yield parsed content items from all configured sources.

        Yields:
            ContentItem: Single JSONL record with LID predictions.
        """
        for input_file in self.infile:
            # Handle S3 transport parameters like in impresso_langident_systems.py
            if input_file.startswith("s3://"):
                transport_params = {"client": self.s3_client}
            else:
                transport_params = {}

            try:
                log.info("Processing file: %s", input_file)
                with smart_open.open(
                    input_file, transport_params=transport_params, encoding="utf-8"
                ) as reader:
                    line_count = 0
                    for line in reader:
                        line_count += 1
                        if line.strip():
                            try:
                                contentitem = json.loads(line)
                                yield contentitem
                            except json.JSONDecodeError as e:
                                log.error(
                                    "JSON decode error in file %s at line %d: %s",
                                    input_file,
                                    line_count,
                                    e,
                                )
                                raise
                log.info(
                    "Successfully processed %d lines from %s", line_count, input_file
                )
            except OSError as e:
                log.error(
                    "Failed to read file %s: %s. "
                    "The file may be corrupted or incomplete.",
                    input_file,
                    e,
                )
                raise RuntimeError(
                    f"Cannot process file {input_file}: {e}. "
                    "The file may be corrupted, incomplete, or in an invalid format."
                ) from e
            except Exception as e:
                log.error(
                    "Unexpected error while processing file %s: %s",
                    input_file,
                    e,
                )
                raise

    def update_lid_distributions(self, content_item: ContentItem) -> None:
        """Record absolute counts for each LID system given one item."""
        for lid in self.lids:
            prediction = extract_prediction(content_item.get(lid))
            if prediction:
                lang, _ = prediction
                self.lid_distributions[lid][lang] += 1.0
                self.lid_absolute_counts[lid][lang] += 1

        orig_lg = content_item.get("orig_lg")
        if isinstance(orig_lg, str) and orig_lg:
            self.lid_distributions["orig_lg"][orig_lg] += 1.0
            self.lid_absolute_counts["orig_lg"][orig_lg] += 1

    def get_votes(self, content_item: ContentItem) -> Optional[VoteTally]:
        """Compute boosted ensemble votes for a content item.

        Args:
            content_item: LID predictions and metadata for one item.

        Returns:
            Optional[VoteTally]: Normalized vote totals or ``None`` when no decision is possible.
        """
        votes: VotesByLanguage = defaultdict(list)

        orig_lg = content_item.get("orig_lg")
        if isinstance(orig_lg, str) and orig_lg:
            votes[orig_lg].append(
                (
                    "orig_lg",
                    self.boost_factor if "orig_lg" in self.boosted_lids else 1.0,
                )
            )

        for lid in self.lids:
            prediction = extract_prediction(content_item.get(lid))
            if not prediction:
                continue
            lang, prob = prediction
            if (
                self.admissible_languages is not None
                and lang not in self.admissible_languages
            ):
                continue
            if prob is None or prob < self.minimal_lid_probability:
                continue
            votes[lang].append(
                (lid, self.boost_factor if lid in self.boosted_lids else 1.0)
            )

        decision: VoteTally = {}
        for lang, votes_lang in votes.items():
            support_count = len(votes_lang)
            total_score = sum(
                vote_score if support_count > 1 else 1.0 for _, vote_score in votes_lang
            )
            if total_score >= self.minimal_vote_score:
                decision[lang] = total_score

        votes_snapshot = {lg: list(records) for lg, records in votes.items()}
        log.debug(
            "Decisions: %s votes=%s content_item=%s",
            decision if decision else None,
            votes_snapshot,
            content_item,
        )

        return decision or None

    def collect_statistics(self) -> None:
        """Accumulate aggregate metrics across all content items."""
        for ci in self.get_next_contentitem():

            # we can infer the newspaper name from impresso content item naming schema
            if self.newspaper is None:
                # the suffix is fixed whereas the former part of the id may vary
                # example of an content item ID: luxzeit1858-1859-01-01-a-i0001
                self.newspaper = ci["id"][0 : len(ci["id"]) - 19]
                log.warning(
                    "Inferred newspaper name from first content item as"
                    f" '{self.newspaper}'"
                )

            # update content type statistics
            content_type = ci.get("tp")
            if isinstance(content_type, str):
                self.contentitem_type_distribution[content_type] += 1
            else:
                self.contentitem_type_distribution["unknown"] += 1

            if content_type == "img":
                continue

            ci_len_value = ci.get("len", 0)
            ci_len = ci_len_value if isinstance(ci_len_value, int) else 0
            self.content_length_stats[ci_len] += 1

            a_ratio_value = ci.get("alphabetical_ratio", 0)
            a_ratio = (
                float(a_ratio_value) if isinstance(a_ratio_value, (int, float)) else 0.0
            )
            if (a_ratio < 0.5) or ci_len * a_ratio < self.minimal_text_length:
                log.debug(f"Ignore short content item: {ci['id']}\t(length: {ci_len})")
                continue

            # update counter for content item with textual content
            self.n += 1

            # update lid systems counts (including orig_lg)
            self.update_lid_distributions(ci)

            # compute the ensemble voting decision (if any)
            decision = self.get_votes(ci)

            lang: Optional[Language] = None
            if decision:
                best_lang, best_score = max(decision.items(), key=lambda item: item[1])
                has_tie = any(
                    abs(candidate_score - best_score) < 1e-9
                    and candidate_lang != best_lang
                    for candidate_lang, candidate_score in decision.items()
                )
                if has_tie:
                    log.warning(
                        f"Ignore decision for {ci['id']} as there is a tie between the"
                        f" two top predicted languages {decision}"
                    )
                else:
                    lang = best_lang
                    log.debug(f"Decision taken: lang={lang} score={best_score}")

            if lang is not None:
                self.lid_distributions["ensemble"][lang] += 1.0
                self.lid_absolute_counts["ensemble"][lang] += 1

            for lid in self.lids:
                lid_prediction = extract_prediction(ci.get(lid))
                if lid_prediction and lang is not None:
                    lid_lang, _ = lid_prediction
                    if lid_lang == lang:
                        self.lg_support[lid][lid_lang] += 1.0

            orig_lg = ci.get("orig_lg")
            if isinstance(orig_lg, str) and orig_lg:
                self.orig_lg_total_decisions += 1
                if lang == orig_lg:
                    self.lg_support["orig_lg"][orig_lg] += 1.0
                elif lang is not None:
                    self.orig_lg_ensemble_disagreements[f"{orig_lg}->{lang}"] += 1
                    log.debug(
                        "Disagreement in %s: orig_lg=%s, ensemble=%s",
                        ci["id"],
                        orig_lg,
                        lang,
                    )

    def compute_support(self) -> None:
        """Convert raw counters into relative support metrics."""

        # Do this before the relative frequencies has been computed
        try:
            orig_lg_n = sum(self.lid_distributions["orig_lg"].values())
            if orig_lg_n > 0.0:
                self.overall_orig_lg_support = round(
                    sum(self.lg_support["orig_lg"].values()) / orig_lg_n,
                    self.round_ndigits,
                )
            else:
                self.overall_orig_lg_support = None
        except ZeroDivisionError:
            self.overall_orig_lg_support = None

        for lid in self.lids.union(["orig_lg"]):
            # if a newspaper has no orig_lg or if none of the predicted outputs of a system got support
            if not self.lg_support.get(lid):
                continue

            # turn support distributions into relative frequencies
            for lang in self.lg_support[lid]:
                self.lg_support[lid][lang] = round(
                    self.lg_support[lid][lang]
                    / max(self.lid_distributions[lid][lang], 1e-12),
                    self.round_ndigits,
                )

        for lid in self.lid_distributions:
            update_relfreq(
                self.lid_distributions[lid], n=self.n, ndigits=self.round_ndigits
            )

        ensemble_distribution = self.lid_distributions["ensemble"]
        self.dominant_language = (
            max(ensemble_distribution, key=ensemble_distribution.get)
            if ensemble_distribution
            else None
        )

        # Log disagreement statistics
        if self.orig_lg_ensemble_disagreements:
            log.info(
                "Found %d disagreements between orig_lg and ensemble decisions",
                sum(self.orig_lg_ensemble_disagreements.values()),
            )
            log.info(
                "Most common disagreements: %s",
                dict(self.orig_lg_ensemble_disagreements.most_common(5)),
            )

    def jsonify(self) -> Dict[str, Any]:
        """Serialize the collected statistics into a JSON-safe dict."""
        json_data = {}

        for attr in self.attrs_for_json:
            json_data[attr] = getattr(self, attr)
            if isinstance(json_data[attr], set):
                json_data[attr] = list(json_data[attr])

        return json_data


def main():
    import argparse

    DESCRIPTION = "Aggregate language-related statistics on content items."

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-l",
        "--log-file",
        dest="log_file",
        help="Write log to FILE",
        metavar="FILE",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--newspaper",
        type=str,
        help="newspaper name for statistics output (default %(default)s)",
    )
    parser.add_argument(
        "--minimal-text-length",
        metavar="n",
        default=200,
        type=int,
        help=(
            "Threshold on article length in chars for computing support (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--boost-factor",
        metavar="B",
        default=1.5,
        type=float,
        help="Boost factor for boosted lids (default %(default)s)",
    )
    parser.add_argument(
        "--minimal-lid-probability",
        metavar="P",
        default=0.25,
        type=float,
        help=(
            "Minimal probability for a LID decision to be considered a vote (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--minimal-vote-score",
        metavar="S",
        default=1.5,
        type=float,
        help=(
            "Minimal vote score from ensemble to reach a decision (default %(default)s)"
        ),
    )
    parser.add_argument(
        "--round-ndigits",
        default=3,
        type=int,
        help="round floats in the output to n digits (default %(default)s)",
    )
    parser.add_argument(
        "--lids",
        nargs="+",
        default=[
            "langdetect",
            "langid",
            "impresso_ft",
            "wp_ft",
            "impresso_langident_pipeline",
            "lingua",
        ],
        choices=[
            "langdetect",
            "langid",
            "impresso_ft",
            "wp_ft",
            "impresso_langident_pipeline",
            "lingua",
        ],
        metavar="LID",
        help=(
            "Names of all LID systems (e.g. langdetect, langid) to use. Do not add"
            " orig_lg here! (default %(default)s)"
        ),
    )
    parser.add_argument(
        "--boosted-lids",
        nargs="+",
        default=[],
        choices=[
            "langdetect",
            "langid",
            "impresso_ft",
            "wp_ft",
            "impresso_langident_pipeline",
            "lingua",
            "orig_lg",
        ],
        metavar="LID",
        help=(
            "Subset of LID systems or orig_lg that are boosted by "
            "a factor if they have support from any other system or orig_lg."
        ),
    )
    parser.add_argument(
        "--admissible-languages",
        nargs="+",
        default=None,
        metavar="L",
        help=(
            "Names of languages considered in the ensemble decisions. "
            "If None, no restrictions are applied (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--git-describe",
        type=str,
        default="",
        help=(
            "output of git describe command for ingesting git version into JSON as"
            " version string"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        metavar="FILE",
        help="write output to FILE (default: stdout)",
    )

    parser.add_argument(
        "infile",
        metavar="INPUT",
        nargs="+",
        type=str,
        help=(
            "Input files of the format jsonl.bz2 or S3 prefix "
            "(s3://BUCKET/PREFIX) to expand to matching files"
        ),
    )

    arguments = parser.parse_args()

    setup_logging(arguments.log_level, arguments.log_file, logger=log)

    log.info("%s", arguments)

    aggregator = AggregatorLID(
        infile=arguments.infile,
        newspaper=arguments.newspaper,
        lids=set(arguments.lids),
        boosted_lids=set(arguments.boosted_lids),
        boost_factor=arguments.boost_factor,
        minimal_vote_score=arguments.minimal_vote_score,
        minimal_lid_probability=arguments.minimal_lid_probability,
        minimal_text_length=arguments.minimal_text_length,
        round_ndigits=arguments.round_ndigits,
        admissible_languages=arguments.admissible_languages,
        git_describe=arguments.git_describe,
        outfile=arguments.outfile,
    )
    aggregator.run()


if __name__ == "__main__":
    main()
