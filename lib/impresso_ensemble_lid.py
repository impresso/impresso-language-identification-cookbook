#!/usr/bin/env python3

"""
Determine the language of an Impresso content item using ensemble decision making.

This module implements an ensemble language identification system that combines
predictions from multiple language identification systems to make final language
decisions for Impresso newspaper content items.

The script takes two intermediate JSON files as input:
1. A JSONLines file with language predictions per content item from various LID systems
2. A JSON file with global statistics and collection-level information

The ensemble decision process includes multiple rules:
- Unequivocal predictions (all systems agree)
- Agreement among off-the-shelf LID systems
- Length-based fallback to dominant collection language
- Weighted voting with confidence scores
- Special handling for original language metadata

Example:
    $ python impresso_ensemble_lid.py \\
        -i predictions.jsonl \\
        -o final_decisions.jsonl \\
        -C newspaper_stats.json \\
        --lids langdetect langid impresso_ft \\
        --validate

"""

__version__ = "2025.06.24"

import copy
import datetime
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Any,
    Tuple,
    TypedDict,
    cast,
)

import jsonschema
import jsonschema.exceptions
from jsonschema.protocols import Validator
import smart_open


from impresso_cookbook import get_s3_client, get_timestamp, read_json, setup_logging

log = logging.getLogger(__name__)


class LidPrediction(TypedDict):
    lang: str
    prob: float


LidPredictions = list[LidPrediction]
LidPredictionMap = dict[str, LidPrediction]


class ImpressoLanguageIdentifierEnsemble:
    """Identify language for each content item using ensemble decision

    :param str infile: JSON file with language predictions per content item.
    :param str outfile: Path to folder where processed JSON files should be saved.
    :param str newspaper_stats_filename: JSON file with aggregated statistics per
        newspaper. Read in into the attribute newspaper_stats
    :param Set[str] lids: Set of LID systems predict to language/probability pairs.
        Therefore, orig_lg is not seen as LID system as it "predicts" only a single
        language if any.
    :param float weight_lb_impresso_ft: voting weight for impresso_ft predicting
        Luxembourgish.
    :param float minimal_lid_probability: Minimal probability for a LID decision to be
        considered a vote.
    :param int minimal_text_length: threshold for text length in characters to apply
        automatic language identification.
    :param float minimal_voting_score: minimal vote score for voting decision to be
        accepted
    :param float threshold_confidence_orig_lg: Ignore original language information when
        below this confidence threshold.
    :param Optional[Set[str]] admissible_languages: Limit languages in the ensemble
        decisions. If None, no restrictions are applied.
    :param Optional[str] diagnostics_json: Filename for diagnostics
    :param bool validate: Validate final lang identification JSON against schema
    :param str git_describe: Output of git describe to use as version if not empty
        string

    :attr list attrs_per_content_item: Defines order of attributes and list of
        attributes to copy over from stage 1 content items' JSON and nullable attributes
        from stage 2
    :attr DefaultDict[Counter] stats: Distribution for any JSON property of interest
        (given as key)
    :attr list results: Collection of content items with their identified language.
    :attr dict schema: JSON schema for the output JSON
    :attr method schema_validator: JSON schema validator
    """

    def __init__(
        self,
        infile: str,
        outfile: str,
        newspaper_stats_filename: str,
        lids: Set[str],
        weight_lb_impresso_ft: float,
        minimal_lid_probability: float,
        minimal_text_length: int,
        minimal_voting_score: float,
        threshold_confidence_orig_lg: float,
        admissible_languages: Optional[Set[str]],
        diagnostics_json: Optional[str],
        validate: bool,
        git_describe: str,
        alphabetical_ratio_threshold: Optional[float] = None,
        dominant_language_threshold: Optional[float] = None,
        exclude_lb: Optional[Set[str]] = None,
    ) -> None:

        self.git_describe: str = git_describe
        self.diagnostics_json: Optional[str] = diagnostics_json

        # Add timing and S3 client support
        self.start_time: Optional[float] = None
        self.s3_client: Any = get_s3_client()
        self.ts: str = get_timestamp()

        self.lids: Set[str] = set(lid for lid in lids if lid != "orig_lg")

        self.attrs_per_content_item: List[Dict[str, Any]] = (
            [
                {"key": "id", "required": True, "source": "language_identifier"},
                {"key": "lg", "required": True},
                {"key": "lg_decision", "required": False},
                {"key": "tp", "required": True, "source": "language_identifier"},
                {"key": "len", "required": True, "source": "language_identifier"},
                {"key": "orig_lg", "required": True, "source": "language_identifier"},
                {
                    "key": "alphabetical_ratio",
                    "required": False,
                    "source": "language_identifier",
                },
                {
                    "key": "impresso_language_identifier_version",
                    "required": False,
                },
                {
                    "key": "language_identifier_version",
                    "required": False,
                    "source": "language_identifier",
                },
                {"key": "year", "required": False},
                {"key": "newspaper", "required": False},
                {"key": "ts", "required": False},
            ]
            + [
                {"key": k, "required": False, "source": "language_identifier"}
                for k in sorted(self.lids)
            ]
            + [
                {"key": "votes", "required": False},
                {"key": "ocrqa", "required": False},
                {"key": "bloom", "required": False},
            ]
        )

        self.infile: str = infile

        self.outfile: str = outfile

        if len(self.lids) < 1:
            log.error("No LID specified. At least one language identifier needed.")
            sys.exit(2)

        self.weight_lb_impresso_ft: float = weight_lb_impresso_ft

        self.admissible_languages: Optional[Set[str]] = (
            set(admissible_languages) if admissible_languages else None
        )

        self.threshold_confidence_orig_lg: float = threshold_confidence_orig_lg
        self.minimal_lid_probability: float = minimal_lid_probability
        self.minimal_text_length: int = minimal_text_length
        self.minimal_voting_score: float = minimal_voting_score
        self.alphabetical_ratio_threshold: float = alphabetical_ratio_threshold or 0.0
        self.dominant_language_threshold: float = dominant_language_threshold or 0.90
        self.exclude_lb: Set[str] = set(exclude_lb) if exclude_lb else set()

        self.schema: Optional[Dict[str, Any]] = None
        self.schema_validator: Optional[Validator] = None
        self.stats: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        # Keys whose value distributions are tracked inside update_stats for diagnostics.
        # Order matters; it defines which per-field counters are populated per result.
        self.stats_keys: List[str] = ["lg", "orig_lg", "tp", "lg_decision"]
        self.newspaper_stats: Dict[str, Any] = read_json(
            newspaper_stats_filename, self.s3_client
        )
        self.results: List[Dict[str, Any]] = []

        self.validate: bool = validate
        if self.validate:
            self.load_schema()

    def run(self) -> None:
        """Run the application.

        This method orchestrates the entire language identification process by:
        1. Processing all content items and making language decisions
        2. Writing the results to the output file
        3. Updating and writing diagnostic statistics
        """

        self.start_time = time.time()

        log.info("Starting ensemble language identification")
        log.info("Input file: %s", self.infile)
        log.info("Output file: %s", self.outfile)
        log.info("Using LID systems: %s", ", ".join(self.lids))

        self.update_impresso_lid_results()
        self.write_output()
        self.update_stats()
        self.write_diagnostics()

        # Log compute time
        total_time = time.time() - self.start_time
        log.info(
            "Ensemble language identification finished in %.2f seconds.", total_time
        )

    def load_schema(self) -> None:
        """
        Load the JSON schema for language identification.

        This method fetches the schema from the specified URL and creates a
        Draft6Validator for it. The schema and the validator are stored as instance
        variables for later use.

        Raises:
            jsonschema.exceptions.SchemaError: If the provided schema is not valid.
            jsonschema.exceptions.RefResolutionError: If the provided schema contains an
            unresolvable JSON reference.
        """
        base_uri = (
            "https://impresso.github.io/impresso-schemas/json/language_identification/"
        )
        schema_file = "language_identification.schema.json"

        with smart_open.open(
            base_uri + schema_file,
            "r",
        ) as f:
            self.schema = json.load(f)

        assert self.schema is not None, "Schema must be loaded before creating resolver"

        resolver = jsonschema.RefResolver(
            referrer=self.schema,
            base_uri=base_uri,
        )
        self.schema_validator = jsonschema.Draft6Validator(
            schema=self.schema,
            resolver=resolver,
        )

    def write_output(self) -> None:
        """Write JSONlines output to the specified output file.

        This method writes all processed language identification results to the
        output file in JSONLines format, where each line contains a complete
        content item with its final language decision.

        If validation is enabled, each result is validated against the schema
        before writing.
        """

        # Handle S3 transport parameters
        if self.outfile.startswith("s3://"):
            transport_params = {"client": self.s3_client}
        else:
            transport_params = {}

        with smart_open.open(
            self.outfile, mode="w", encoding="utf-8", transport_params=transport_params
        ) as of:
            for result in self.results:
                if self.validate and self.schema_validator:
                    try:
                        self.schema_validator.validate(result)
                    except jsonschema.exceptions.ValidationError as e:
                        log.error(
                            "Validation error for content item %s: %s",
                            result.get("id", "unknown"),
                            e.message,
                        )
                        raise
                print(json.dumps(result, ensure_ascii=False), file=of)

    def write_diagnostics(self) -> None:
        """Write JSON diagnostics with per-newspaper stats.

        This method writes diagnostic information including statistics and metadata
        about the language identification process to the specified diagnostics file
        in JSON format.
        """

        if self.diagnostics_json:
            # Handle S3 transport parameters
            if self.diagnostics_json.startswith("s3://"):
                transport_params = {"client": self.s3_client}
            else:
                transport_params = {}

            with smart_open.open(
                self.diagnostics_json,
                mode="w",
                encoding="utf-8",
                transport_params=transport_params,
            ) as of:
                print(json.dumps(self.stats), file=of)

    def next_content_item(self) -> Iterable[Dict[str, Any]]:
        """Yield next content item from the input file.

        This generator function reads the input JSONLines file and yields
        each content item as a dictionary for processing.

        :return: Iterator over content item dictionaries.
        :rtype: Iterable[Dict[str, Any]]
        """

        with smart_open.open(
            self.infile,
            mode="r",
            encoding="utf-8",
            transport_params=(
                {"client": self.s3_client} if self.infile.startswith("s3://") else {}
            ),
        ) as reader:
            for line in reader:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def cleanup_attrs(
        self, jinfo: Dict[str, Any], content_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return copy of jinfo with ordered required attributes.

        Attributes with None value that are not required are not copied over.
        Extracts OCR QA information for the decided language if available.

        :param Dict[str, Any] jinfo: Content item dictionary to clean up.
        :param Dict[str, Any] content_item: Original content item with OCR QA data.
        :return: Cleaned content item dictionary with ordered attributes.
        :rtype: Dict[str, Any]
        """
        # Extract OCR QA info for the decided language if lg is set
        if "lg" in jinfo and jinfo["lg"]:
            ocrqa_score, bloom_ref = self.extract_ocrqa_for_language(
                content_item, jinfo["lg"]
            )
            if ocrqa_score is not None:
                jinfo["ocrqa"] = ocrqa_score
            if bloom_ref is not None:
                jinfo["bloom"] = bloom_ref

        result = {}
        for a in self.attrs_per_content_item:
            a_key = a["key"]
            if a.get("required"):
                result[a_key] = jinfo.get(a_key)
            elif jinfo.get(a_key) is not None:
                result[a_key] = jinfo[a_key]
        return result

    def extract_ocrqa_for_language(
        self, content_item: Dict[str, Any], language: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Extract OCR QA score and bloom filter reference for a given language.

        :param Dict[str, Any] content_item: Content item with potential ocrqa data
        :param str language: Language code to extract OCR QA info for
        :return: Tuple of (ocrqa_score, bloom_reference) or (None, None)
        :rtype: Tuple[Optional[float], Optional[str]]
        """
        if "ocrqa" not in content_item or not content_item["ocrqa"]:
            return None, None

        ocrqa_data = content_item["ocrqa"]
        if not isinstance(ocrqa_data, dict) or language not in ocrqa_data:
            return None, None

        lang_ocrqa = ocrqa_data[language]
        if not isinstance(lang_ocrqa, dict):
            return None, None

        # Extract score
        score = lang_ocrqa.get("score")
        if score is not None and isinstance(score, (int, float)):
            # Ensure score is within valid range [0.0, 1.0]
            score = max(0.0, min(1.0, float(score)))
        else:
            score = None

        # Extract model_id and construct bloom filter reference
        model_id = lang_ocrqa.get("model_id")
        bloom = model_id or None

        return score, bloom

    def get_best_lid(self, jinfo: Dict[str, Any]) -> LidPredictionMap:
        """Extract the top prediction from each LID system.

        For each language identification system, this method extracts only the
        highest-confidence prediction, discarding any additional predictions.

        :param Dict[str, Any] jinfo: Content item dictionary with LID predictions.
        :return: Dictionary mapping LID system names to their top predictions.
        :rtype: Dict[str, Dict[str, Union[str, float]]]
        """
        result: LidPredictionMap = {}
        for lid_system in self.lids:
            lid_preds = cast(Optional[LidPredictions], jinfo.get(lid_system))
            if lid_preds:
                result[lid_system] = lid_preds[0]
        return result

    def get_votes(self, content_item: Dict[str, Any]) -> Dict[str, float]:
        """Return dictionary with weighted votes per language.

        This method calculates the weighted votes for each language based on the
        predictions from various language identification systems (LIDs). It applies
        filters for admissible languages, minimal probability thresholds, and boosts
        votes based on predefined confidence levels.

        :param Dict[str, Any] content_item: A dictionary representing a single content item
            with LID predictions.
        :return: A dictionary containing the weighted votes for each language.
        :rtype: Dict[str, float]
        """

        # Check if alphabetical_ratio is below the threshold
        if (
            content_item.get("alphabetical_ratio", 1.0)
            < self.alphabetical_ratio_threshold
        ):
            log.debug(
                "Content item %s: Alphabetical ratio %s below threshold %s, using"
                " dominant language",
                content_item["id"],
                content_item.get("alphabetical_ratio", 1.0),
                self.alphabetical_ratio_threshold,
            )
            return {self.newspaper_stats["dominant_language"]: 1.0}

        # Initialize a dictionary to store votes for each language
        votes: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
        log.debug("Content item %s: Starting vote calculation", content_item["id"])

        # Iterate over each LID system to collect votes
        for lid in self.lids:

            # Check if the LID system has predictions for the content item
            if (
                lid in content_item
                and content_item[lid] is not None
                and len(content_item[lid]) > 0
            ):
                lid_predictions = cast(Optional[LidPredictions], content_item.get(lid))
                if lid_predictions:
                    top_prediction = lid_predictions[0]
                    lang, prob = top_prediction["lang"], float(top_prediction["prob"])
                    log.debug(
                        "Content item %s: %s predicts %s with probability %s",
                        content_item["id"],
                        lid,
                        lang,
                        prob,
                    )

                    # Filter predictions based on admissible languages
                    if (
                        self.admissible_languages is None
                        or lang in self.admissible_languages
                    ):
                        # Check if this newspaper should exclude lb language
                        newspaper_id = content_item["id"][
                            0 : len(content_item["id"]) - 19
                        ]
                        if lang == "lb" and newspaper_id in self.exclude_lb:
                            log.debug(
                                "Content item %s: %s prediction of %s excluded for"
                                " newspaper %s",
                                content_item["id"],
                                lid,
                                lang,
                                newspaper_id,
                            )
                            continue

                        # Filter predictions based on minimal probability threshold
                        if prob >= self.minimal_lid_probability:
                            lang_support = (
                                self.newspaper_stats["lg_support"][lid].get(lang) or 0.0
                            )
                            log.debug(
                                "Content item %s: %s language support for %s: %s",
                                content_item["id"],
                                lid,
                                lang,
                                lang_support,
                            )

                            # Initialize vote_score to 0.0
                            vote_score = 0.0

                            # Calculate the vote score based on confidence levels
                            if lang_support:
                                vote_score = prob * lang_support

                                # Check if newspaper has strong dominance and this is not the dominant language
                                dominant_lang = self.newspaper_stats[
                                    "dominant_language"
                                ]
                                dominant_lang_ratio = self.newspaper_stats.get(
                                    "dominant_language_ratio", 0.0
                                )

                                if (
                                    dominant_lang_ratio
                                    >= self.dominant_language_threshold
                                    and lang != dominant_lang
                                ):
                                    # Apply penalty for non-dominant languages in highly dominant newspapers
                                    dominance_penalty = 1.0 - (
                                        dominant_lang_ratio
                                        - self.dominant_language_threshold
                                    ) / (1.0 - self.dominant_language_threshold)
                                    original_score = vote_score
                                    vote_score *= dominance_penalty
                                    log.debug(
                                        "Content item %s: Applied dominance penalty to"
                                        " %s for %s: %s * %s = %s (dominant lang: %s,"
                                        " ratio: %s)",
                                        content_item["id"],
                                        lid,
                                        lang,
                                        original_score,
                                        dominance_penalty,
                                        vote_score,
                                        dominant_lang,
                                        dominant_lang_ratio,
                                    )

                            log.debug(
                                "Content item %s: %s initial vote score for %s: %s "
                                "(prob %s * support %s)",
                                content_item["id"],
                                lid,
                                lang,
                                vote_score,
                                prob,
                                lang_support,
                            )

                            # Apply special weight for impresso_ft predicting Luxembourgish
                            if lid == "impresso_ft" and lang == "lb":
                                original_score = vote_score
                                vote_score *= self.weight_lb_impresso_ft
                                log.debug(
                                    "Content item %s: Applied Luxembourgish boost to"
                                    " %s: %s * %s = %s",
                                    content_item["id"],
                                    lid,
                                    original_score,
                                    self.weight_lb_impresso_ft,
                                    vote_score,
                                )

                            # Append the vote score to the list for the language
                            votes[lang].append((lid, vote_score))
                            log.debug(
                                "Content item %s: Added vote for %s from %s: %s",
                                content_item["id"],
                                lang,
                                lid,
                                vote_score,
                            )
                        else:
                            log.debug(
                                "Content item %s: %s - "
                                "No language support for %s, vote rejected",
                                content_item["id"],
                                lid,
                                lang,
                            )
                    else:
                        log.debug(
                            "Content item %s: %s language %s "
                            "not in admissible languages, vote rejected",
                            content_item["id"],
                            lid,
                            lang,
                        )
                else:
                    log.debug(
                        "Content item %s: No predictions from %s",
                        content_item["id"],
                        lid,
                    )

        # Aggregate the vote scores for each language
        decision: Dict[str, float] = {}
        for lang in votes:
            total_score = sum(vote_score for (_, vote_score) in votes[lang])
            decision[lang] = total_score
            contributing_lids = [lid for (lid, _) in votes[lang]]
            log.debug(
                "Content item %s: Total vote for %s: %s from systems: %s",
                content_item["id"],
                lang,
                total_score,
                contributing_lids,
            )

        if decision:
            sorted_decision = sorted(
                decision.items(), key=lambda item: item[1], reverse=True
            )
            log.debug(
                "Content item %s: Final vote scores: %s",
                content_item["id"],
                sorted_decision,
            )
        else:
            log.debug("Content item %s: No votes collected", content_item["id"])
        return decision

    def update_impresso_lid_results(self) -> None:
        """Update self.results with all language classification decisions.

        This method processes each content item from the input file and makes
        language identification decisions, storing the results in self.results.
        """

        for c in self.next_content_item():
            log.info("Processing %s", c["id"])
            self.results.append(self.decide_lg(c))

    def decide_lg(self, content_item: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict with decision information for a content item.

        This method applies the ensemble decision rules to determine the final
        language for a content item. It handles various scenarios including
        image content, unequivocal predictions, length-based decisions, and
        weighted voting.

        :param Dict[str, Any] content_item: Content item with language predictions.
        :return: Content item with final language decision and metadata.
        :rtype: Dict[str, Any]
        """

        decided_content_item: Dict[str, Any] = {}

        # copy relevant attributes from stage 1 for each content item
        for d in self.attrs_per_content_item:
            if d.get("source") == "language_identifier":
                decided_content_item[d["key"]] = copy.copy(content_item.get(d["key"]))

        content_id: Optional[str] = decided_content_item.get("id")
        decided_content_item["newspaper"] = (
            content_id[0 : len(content_id) - 19] if content_id else ""
        )
        decided_content_item["year"] = content_id[-18:-14] if content_id else ""
        decided_content_item["ts"] = self.ts
        decided_content_item.update(
            {
                "impresso_language_identifier_version": {
                    "version": self.git_describe or __version__,
                    "ts": (
                        datetime.datetime.now(datetime.timezone.utc).isoformat(
                            sep="T", timespec="seconds"
                        )
                    ),
                }
            }
        )

        if decided_content_item["tp"] == "img":
            return self.cleanup_attrs(decided_content_item, content_item)

        trust_orig_lg = False
        if overall_orig_lg_support := self.newspaper_stats.get(
            "overall_orig_lg_support"
        ):
            trust_orig_lg = overall_orig_lg_support > self.threshold_confidence_orig_lg

        log.debug(
            "Content item %s: Original language trust check - "
            "overall_orig_lg_support: %s, "
            "threshold: %s, "
            "trust_orig_lg: %s",
            content_item["id"],
            overall_orig_lg_support,
            self.threshold_confidence_orig_lg,
            trust_orig_lg,
        )

        dominant_lg = self.newspaper_stats["dominant_language"]
        log.debug(
            "Content item %s: Dominant language: %s", content_item["id"], dominant_lg
        )

        # rule 1: ignore original language information when not trustworthy
        if not trust_orig_lg or not content_item.get("orig_lg"):
            log.debug(
                "Content item %s: Rule 1 - "
                "Ignoring original language (trust_orig_lg: %s, "
                "orig_lg present: %s)",
                content_item["id"],
                trust_orig_lg,
                bool(content_item.get("orig_lg")),
            )
            content_item["orig_lg"] = None
            self.lids.discard("orig_lg")
        else:
            # set confidence value of original language information as probability
            # the original probability was always 1 before
            orig_lg_support = self.newspaper_stats["lg_support"]["orig_lg"].get(
                content_item["orig_lg"], 0.00001
            )
            original_lang = cast(str, content_item["orig_lg"])
            log.debug(
                "Content item %s: Rule 1 - Using original language %s with support %s",
                content_item["id"],
                original_lang,
                orig_lg_support,
            )
            content_item["orig_lg"] = [
                {"lang": original_lang, "prob": float(orig_lg_support)}
            ]

        # rule 2
        all_lid_preds: LidPredictionMap = self.get_best_lid(content_item)
        all_lid_languages: Set[str] = {pred["lang"] for pred in all_lid_preds.values()}

        log.debug(
            "Content item %s: All LID predictions: %s",
            content_item["id"],
            all_lid_preds,
        )
        log.debug(
            "Content item %s: All predicted languages: %s",
            content_item["id"],
            all_lid_languages,
        )

        # rule 2a: follow unequivocal predictions
        if len(all_lid_languages) == 1:
            decided_language = min(all_lid_languages)
            log.debug(
                "Content item %s: Rule 2a - All systems agree on language: %s",
                content_item["id"],
                decided_language,
            )
            decided_content_item["lg"] = decided_language
            decided_content_item["lg_decision"] = "all"
            return self.cleanup_attrs(decided_content_item, content_item)

        all_but_impresso_ft_lid_languages = set(
            str(all_lid_preds[lid]["lang"])
            for lid in all_lid_preds
            if lid != "impresso_ft"
        )

        # rule 2b: off-the-shelf LID agree on language other than DE or FR
        if len(all_but_impresso_ft_lid_languages) == 1:
            other_lg: str = next(
                iter(all_but_impresso_ft_lid_languages)
            )  # More explicit than min()
            log.debug(
                "Content item %s: Rule 2b - All non-impresso_ft systems agree on: %s",
                content_item["id"],
                other_lg,
            )

            text_length_condition = (
                content_item["len"] * content_item["alphabetical_ratio"]
                >= self.minimal_text_length
            )
            in_ensemble_distribution = (
                other_lg in self.newspaper_stats["lid_distributions"]["ensemble"]
            )
            is_non_major_language = other_lg not in {"de", "fr", "en", "it"}

            log.debug(
                "Content item %s: Rule 2b conditions - non-major"
                " language: %s, in ensemble distribution:"
                " %s, text length sufficient:"
                " %s (len=%s,"
                " alpha_ratio=%s,"
                " threshold=%s)",
                content_item["id"],
                is_non_major_language,
                in_ensemble_distribution,
                text_length_condition,
                content_item["len"],
                content_item["alphabetical_ratio"],
                self.minimal_text_length,
            )

            if (
                is_non_major_language
                and in_ensemble_distribution
                and text_length_condition
            ):
                log.debug(
                    "Content item %s: Rule 2b accepted - language: %s",
                    content_item["id"],
                    other_lg,
                )
                decided_content_item["lg"] = other_lg
                decided_content_item["lg_decision"] = "all-but-impresso_ft"
                return self.cleanup_attrs(decided_content_item, content_item)
            else:
                log.debug(
                    "Content item %s: Rule 2b rejected for %s",
                    content_item["id"],
                    other_lg,
                )

        # rule 2c: set dominant language of newspaper for very short articles
        text_len: int = decided_content_item.get("len", 0)
        if text_len and text_len < self.minimal_text_length:
            log.debug(
                "Content item %s: Rule 2c - Text too short"
                " (%s < %s), using"
                " dominant language: %s",
                content_item["id"],
                text_len,
                self.minimal_text_length,
                dominant_lg,
            )
            decided_content_item["lg"] = dominant_lg
            decided_content_item["lg_decision"] = "dominant-by-len"
            return self.cleanup_attrs(decided_content_item, content_item)

        votes: Dict[str, float] = self.get_votes(content_item)
        sorted_votes: List[Tuple[str, float]] = sorted(
            votes.items(), key=lambda item: item[1], reverse=True
        )
        decided_content_item["votes"] = [
            {"lang": lang, "vote": round(score, 3)} for lang, score in sorted_votes
        ]

        log.debug(
            "Content item %s: Vote results: %s",
            content_item["id"],
            decided_content_item["votes"],
        )

        if not sorted_votes:
            log.debug(
                "Content item %s: No votes received, using dominant language: %s",
                content_item["id"],
                dominant_lg,
            )
            decided_content_item["lg"] = dominant_lg
            decided_content_item["lg_decision"] = "dominant-by-lowvote"
            return self.cleanup_attrs(decided_content_item, content_item)

        best_vote_score: float = sorted_votes[0][1]
        if best_vote_score < self.minimal_voting_score:
            log.debug(
                "Content item %s: Best vote score %s "
                "below threshold %s, "
                "using dominant language: %s",
                content_item["id"],
                best_vote_score,
                self.minimal_voting_score,
                dominant_lg,
            )
            decided_content_item["lg"] = dominant_lg
            decided_content_item["lg_decision"] = "dominant-by-lowvote"
            return self.cleanup_attrs(decided_content_item, content_item)

        # rule 3: get decision by ensemble voting for less obvious cases
        winning_language: str = sorted_votes[0][0]
        log.debug(
            "Content item %s: Rule 3 - Voting decision: %s with score %s",
            content_item["id"],
            winning_language,
            best_vote_score,
        )
        decided_content_item["lg"] = winning_language
        decided_content_item["lg_decision"] = "voting"
        return self.cleanup_attrs(decided_content_item, content_item)

    def update_stats(self) -> None:
        """Update per-newspaper statistics for diagnostics.

        Iterates over finalized language-identification results and aggregates:
            * per-field counters for each key in ``self.stats_keys`` (e.g., ``lg``,
              ``orig_lg``) using their recorded values in ``self.results``.
            * per-newspaper/year totals stored in ``self.stats['N']`` keyed by the
              combination ``"<newspaper>-<year>"`` to support temporal diagnostics.

        The counters in ``self.stats`` are mutated in place and later exported via
        :meth:`write_diagnostics`.
        """

        for r in self.results:
            for p in self.stats_keys:
                if (v := r.get(p)) is not None:
                    self.stats[p][v] += 1
            self.stats["N"][f'{self.newspaper_stats["newspaper"]}-{r["year"]}'] += 1


def main() -> None:
    """Main function to run the Impresso Language Identifier Ensemble."""
    import argparse

    DESCRIPTION = (
        "Classify language of impresso content items given all collected evidence"
    )

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
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
        "-C",
        "--newspaper-stats-filename",
        type=str,
        required=True,
        help="newspaper statistics JSON file",
    )

    parser.add_argument(
        "--threshold_confidence_orig_lg",
        default=0.75,
        type=float,
        help="ignore original language when below this threshold (default %(default)s)",
    )

    parser.add_argument(
        "-i",
        "--infile",
        required=True,
        help="path to input file from s3 batch, json format",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        required=True,
        help="path to folder where processed .json files should be saved",
    )
    parser.add_argument(
        "--weight-lb-impresso-ft",
        metavar="W",
        default=3,
        type=float,
        help=(
            "special voting weight for impresso_ft predicting Luxembourgish (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--minimal-lid-probability",
        metavar="P",
        default=0.5,
        type=float,
        help=(
            "minimal probability for a LID decision to be considered a vote (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--minimal-voting-score",
        metavar="W",
        default=0.5,
        type=float,
        help=(
            "minimal vote score for voting decision to be accepted (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "validate final lang identification JSON against schema (default"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--diagnostics-json",
        type=str,
        help="filename for statistical diagnostics information in JSON format",
    )
    parser.add_argument(
        "-m",
        "--minimal-text-length",
        default=20,
        type=int,
        help=(
            "minimal text length of content items to apply automatic language"
            " identification (default %(default)s)"
        ),
    )
    parser.add_argument(
        "--lids",
        nargs="+",
        default=[],
        metavar="LID",
        help=(
            "names of all LID systems (e.g. langdetect, langid) to use. Do not add"
            " orig_lg here!"
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
        help="git describe output for ingesting version into JSON as version string",
    )
    parser.add_argument(
        "--alphabetical-ratio-threshold",
        default=0.5,
        type=float,
        help=(
            "threshold for alphabetical ratio below which dominant language is selected"
            " (default %(default)s)"
        ),
    )
    parser.add_argument(
        "--dominant-language-threshold",
        default=0.90,
        type=float,
        help=(
            "threshold for dominant language ratio above which non-dominant languages "
            "receive penalty in voting (default %(default)s)"
        ),
    )
    parser.add_argument(
        "--exclude-lb",
        nargs="+",
        default=[],
        metavar="NEWSPAPER",
        help=(
            "newspaper acronyms for which Luxembourgish (lb) language predictions "
            "should be excluded (default: %(default)s)"
        ),
    )

    arguments = parser.parse_args()

    setup_logging(arguments.log_level, arguments.log_file, logger=log)

    log.info("%s", arguments)

    try:
        ensemble = ImpressoLanguageIdentifierEnsemble(
            infile=arguments.infile,
            outfile=arguments.outfile,
            newspaper_stats_filename=arguments.newspaper_stats_filename,
            lids=set(arguments.lids),
            weight_lb_impresso_ft=arguments.weight_lb_impresso_ft,
            minimal_lid_probability=arguments.minimal_lid_probability,
            minimal_text_length=arguments.minimal_text_length,
            threshold_confidence_orig_lg=arguments.threshold_confidence_orig_lg,
            minimal_voting_score=arguments.minimal_voting_score,
            admissible_languages=arguments.admissible_languages,
            diagnostics_json=arguments.diagnostics_json,
            git_describe=arguments.git_describe,
            validate=arguments.validate,
            alphabetical_ratio_threshold=arguments.alphabetical_ratio_threshold,
            dominant_language_threshold=arguments.dominant_language_threshold,
            exclude_lb=set(arguments.exclude_lb),
        )
        ensemble.run()
    except Exception as e:
        log.error("Ensemble language identification failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Processing error: %s", e, exc_info=True)
        sys.exit(2)
