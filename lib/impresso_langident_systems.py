#!/usr/bin/env python3

"""
Language identification module for newspaper content items.

This module provides a flexible framework for running multiple language identification
(LID) tools in parallel on the same content items. It allows comparison and analysis
of results from different LID systems including:

- langdetect: Statistical language detection using character n-grams
- langid: Language identification using n-gram features
- FastText models: Including custom impresso and Wikipedia-trained models
- impresso_langident_pipeline: Advanced language identification using the impresso pipeline
- lingua: High-accuracy language detection library

The module supports multiple input formats:
- **Rebuilt format**: Traditional impresso content item format (JSONL with content items)
- **Canonical format**: Impresso canonical page schema format (requires issue metadata file)

Additional capabilities:
- **OCR Quality Assessment**: Optional evaluation of OCR quality for all supported languages
  using the impresso_pipelines.ocrqa module

The module uses a dynamic registry pattern to easily configure which LID systems to run
and handles text validation, model initialization, and result aggregation in a modular way.
All configured LID systems are applied to each content item, and their results are stored
side-by-side in the output for comparison.

Key features:
- Parallel execution of multiple LID systems on the same content items
- Support for both rebuilt and canonical page formats
- Configurable text length and alphabetical ratio thresholds
- Support for variable number of LID models (from 1 to all available)
- Robust error handling for individual models
- Detection and logging of disagreements between LID systems
- Optional OCR quality assessment across multiple languages
- S3 and local file support for input/output
- Comprehensive logging with structured output and statistics

Example usage:
    # Compare results from multiple LID systems on rebuilt format:
    processor = LanguageIdentifier(
        infile="input.jsonl",
        outfile="output.jsonl",
        lids=["langdetect", "langid", "impresso_ft", "wp_ft"],
        minimal_text_length=20,
        alphabetical_ratio_threshold=0.0  # Default: no alphabetical ratio filtering
    )
    processor.run()

    # Use canonical format with OCR quality assessment:
    processor = LanguageIdentifier(
        infile="s3://bucket/NEWSPAPER/pages/NEWSPAPER-YEAR/NEWSPAPER-",
        outfile="output.jsonl",
        lids=["impresso_langident_pipeline", "lingua"],
        format="canonical",
        issue_file="s3://bucket/NEWSPAPER/issues/NEWSPAPER-YEAR.issue.jsonl.bz2",
        ocrqa=True,
        minimal_text_length=20
    )
    processor.run()
"""

__version__ = "2025.06.21"

import json
import logging
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Iterable, Set, Union, Tuple

import fasttext
import langdetect
from langdetect.lang_detect_exception import LangDetectException
from langid import langid
import smart_open

try:
    from impresso_pipelines.langident import LangIdentPipeline

    IMPRESSO_LANGIDENT_PIPELINE_AVAILABLE = True
except ImportError:
    IMPRESSO_LANGIDENT_PIPELINE_AVAILABLE = False

try:
    from impresso_pipelines.ocrqa import OCRQAPipeline

    IMPRESSO_OCRQA_AVAILABLE = True
except ImportError:
    IMPRESSO_OCRQA_AVAILABLE = False

try:
    from lingua import LanguageDetectorBuilder, Language, IsoCode639_1

    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False


from impresso_cookbook import (
    get_s3_client,
    get_timestamp,
    yield_s3_objects,
    setup_logging,
)

log = logging.getLogger(__name__)

# Log warning if impresso_pipelines is not available
if not IMPRESSO_LANGIDENT_PIPELINE_AVAILABLE:
    log.warning(
        "impresso_pipelines package not available - impresso_langident_pipeline will"
        " not be functional"
    )
    log.warning(
        "Please install it with 'pip install impresso_pipelines' to use this feature."
    )

# Log warning if impresso_ocrqa is not available
if not IMPRESSO_OCRQA_AVAILABLE:
    log.warning(
        "impresso_pipelines.ocrqa package not available - ocrqa will not be functional"
    )
    log.warning(
        "Please install it with 'pip install impresso_pipelines' to use this feature."
    )


def alphabetical_ratio(text: str) -> float:
    """Return the percentage of alphabetic characters of a text."""
    if not text:
        return 0.0
    filtered_length = len(re.sub(r"[\W_\d]+", "", text))
    return filtered_length / len(text) if filtered_length else 0.0


def average_distribution(
    listoflist: List[List], round_ndigits: int = 9
) -> List[Dict[str, Union[str, float]]]:
    """Return dictionary of averaged probabilities per language.

    :param int round_ndigits: Number of decimal places for probabilities
    :param List[List] listoflist: Results of multiple language identification.
    :return: Dictionary with the averaged probabilities per language
    :rtype: List[Dict[str, float]]

    """

    total = len(listoflist)
    counter = Counter()
    for row in listoflist:
        for r in row:
            counter[r.lang] += r.prob
    for lang in counter:
        counter[lang] = counter[lang] / total

    result = [
        {"lang": lang, "prob": round(prob, round_ndigits)}
        for lang, prob in counter.most_common()
    ]

    log.debug(
        "DEBUG-LANGDETECT-DIVERSITY Length: %s Predictions: %s",
        len(listoflist),
        listoflist,
    )

    return result


def avg_langdetect_lid(
    text: str,
    n: int,
    threshold: float = 0.95,
    seed: int = 42,
    default_languages: Tuple[str] = ("de", "fr"),
    round_ndigits: int = 9,
) -> List[Dict[str, Union[str, float]]]:
    """Compute averaged lid score from n samples using Langdetect.

    For efficiency, drawing stops if the top-most language has a higher probability than
    threshold

    :param int round_ndigits: Number of decimal places for probabilities.
    :param str text: Text to classify.
    :param int n: Number of samples.
    :param int seed: Initial random seed for langdetect
    :param Set[str] default_languages: Set of language where early stopping is allowed
        for highly probably languages
    :param float threshold: Threshold for early-stopping of sampling.
    :return: Dictionary with the averaged probabilities per language
    :rtype: List[Dict[str, float]]

    """
    langdetect.DetectorFactory.seed = seed

    results = []
    text = text.lower()  # add lower case text to increase detection probability
    for i in range(n):
        langdetect.DetectorFactory.seed += i
        result = langdetect.detect_langs(text)
        results.append(result)
        if result[0].prob > threshold and result[0].lang in default_languages:
            break

    return average_distribution(results, round_ndigits)


def fasttext_lid(
    text: str, ft_model, round_ndigits: int = 3
) -> List[Dict[str, Union[str, float]]]:
    """
    Return results of a fasttext model.

    The only normalization is mapping digits to 0. The internal function predict of
    fasttext returns a pair of tuples

    In [16]: m.predict(''' l'eût cru, le rêve de M. Mitterand, c'est d'e''',k=3)
    Out[16]: (('__label__fr', '__label__lb', '__label__de'),
             array([9.99996185e-01, 2.38023513e-05, 1.00000034e-05]))
    """

    # ignore digits
    text = re.sub(r"\d+", "", text)

    labels, probs = ft_model.predict(text, k=5, threshold=0.05)
    result = [
        {
            "lang": lang.replace("__label__", ""),
            "prob": float(min(1, round(probs[i], round_ndigits))),
        }
        for (i, lang) in enumerate(labels)
    ]

    return result


class ImpressoLanguageIdentifierSystems(object):
    """Apply multiple language identification systems to content items.

    This class runs multiple LID systems in parallel on the same content items
    for comparison and analysis.
    """

    def __init__(
        self,
        infile: str,
        outfile: str,
        impresso_ft: str,
        wp_ft: str,
        minimal_text_length: int,
        lids: list,
        round_ndigits: int,
        git_describe: str,
        alphabetical_ratio_threshold: float,
        format: str = "rebuilt",
        debug: bool = False,
        issue_file: str = None,
        ocrqa: bool = False,
    ):

        self.infile: str = infile
        self.outfile: str = outfile
        self.impresso_ft: str = impresso_ft
        self.wp_ft: str = wp_ft
        self.minimal_text_length: int = minimal_text_length
        self.format: str = format
        self.debug: bool = debug
        self.issue_file: str = issue_file
        self.ocrqa: bool = ocrqa

        # Validate that issue_file is provided for canonical format
        if self.format == "canonical" and not self.issue_file:
            raise ValueError("issue_file must be provided when using canonical format")

        # Validate that impresso_pipelines.ocrqa is available if ocrqa is requested
        if self.ocrqa and not IMPRESSO_OCRQA_AVAILABLE:
            raise ValueError(
                "impresso_pipelines.ocrqa is not available but --ocrqa was requested"
            )

        self.lids: Set[str] = set(lids)
        log.info(
            "Predicting with the following off-the-shelve LID systems: %s.",
            ", ".join(lids),
        )
        self.round_ndigits = round_ndigits
        self.git_describe = git_describe
        self.s3_client = get_s3_client()
        self.results = []
        self.alphabetical_ratio_threshold = alphabetical_ratio_threshold
        self.start_time = None
        self.ts = get_timestamp()
        self.stats = {
            "processed_items": 0,
            "skipped_no_text": 0,
            "skipped_short_text": 0,
            "skipped_low_alpha": 0,
            "language_identified": 0,
            "language_disagreements": 0,
        }

        # OCR QA statistics if enabled
        if self.ocrqa:
            self.stats["ocrqa_processed"] = 0
            self.stats["ocrqa_failed"] = 0
            self.stats["ocrqa_max_languages"] = (
                {}
            )  # Counter for languages with max OCR QA scores

    def run(self):
        """Run the language identification process."""
        self.start_time = time.time()

        log.info(
            "Starting language identification process for input file: %s", self.infile
        )
        log.info("Output will be written to: %s", self.outfile)
        log.info("Using LID systems: %s", ", ".join(self.lids))

        self.language_identification()
        self.write_output()

        # Log statistics
        self._log_statistics()

        # Log compute time
        total_time = time.time() - self.start_time
        log.info(
            "Language identification finished for %s in %.2f seconds.",
            self.infile,
            total_time,
        )

    def _initialize_models(self):
        """Initialize language identification models based on requested LID systems."""
        models = {}

        log.info("Initializing models for input file: %s", self.infile)

        # Define model initializers
        model_initializers = {
            "langid": lambda: langid.LanguageIdentifier.from_modelstring(
                langid.model, norm_probs=True
            ),
            "impresso_ft": lambda: (
                fasttext.load_model(self.impresso_ft) if self.impresso_ft else None
            ),
            "wp_ft": lambda: fasttext.load_model(self.wp_ft) if self.wp_ft else None,
            "impresso_langident_pipeline": lambda: (
                LangIdentPipeline() if IMPRESSO_LANGIDENT_PIPELINE_AVAILABLE else None
            ),
            "lingua": lambda: (
                LanguageDetectorBuilder.from_all_languages().build()
                if LINGUA_AVAILABLE
                else None
            ),
        }

        # Initialize OCR QA pipeline if requested
        if self.ocrqa:
            try:
                models["ocrqa"] = OCRQAPipeline()
                log.info("Successfully loaded OCR QA pipeline for %s", self.infile)
            except Exception as e:
                log.error("Failed to load OCR QA pipeline for %s: %s", self.infile, e)

        # Initialize only requested models
        for lid_system in self.lids:
            if lid_system in model_initializers:
                try:
                    model = model_initializers[lid_system]()
                    if model is not None:
                        models[lid_system] = model
                        log.info(
                            "Successfully loaded %s model for %s",
                            lid_system,
                            self.infile,
                        )
                    else:
                        log.warning(
                            "Model path not provided for %s when processing %s",
                            lid_system,
                            self.infile,
                        )
                        if (
                            lid_system == "impresso_langident_pipeline"
                            and not IMPRESSO_LANGIDENT_PIPELINE_AVAILABLE
                        ):
                            log.warning(
                                "impresso_pipelines package not available for %s",
                                self.infile,
                            )
                        if lid_system == "lingua" and not LINGUA_AVAILABLE:
                            log.warning(
                                "lingua package not available for %s", self.infile
                            )
                except Exception as e:
                    log.error(
                        "Failed to load %s model for %s: %s", lid_system, self.infile, e
                    )
            elif (
                lid_system != "langdetect"
            ):  # langdetect doesn't need model initialization
                log.warning(
                    "Unknown LID system %s when processing %s", lid_system, self.infile
                )

        return models

    def _apply_langdetect(
        self, text: str
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Apply langdetect language identification."""
        try:
            return avg_langdetect_lid(text, 3, round_ndigits=self.round_ndigits)
        except LangDetectException:
            log.error(
                "LANGDETECT-ERROR for %s with text: %s %s",
                self.infile,
                text,
                sys.exc_info()[0],
            )
            return None

    def _apply_langid(
        self, text: str, model
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Apply langid language identification."""
        try:
            lang_orig, lang_prob_orig = model.classify(text.lower())
            return [
                {
                    "lang": lang_orig,
                    "prob": round(lang_prob_orig, self.round_ndigits),
                }
            ]
        except Exception:
            log.error("LANGID-ERROR for %s: %s", self.infile, sys.exc_info()[0])
            return None

    def _apply_fasttext(
        self, text: str, model, model_name: str
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Apply FastText language identification."""
        try:
            return fasttext_lid(text, model, round_ndigits=self.round_ndigits)
        except Exception:
            log.error(
                "%s-ERROR for %s: %s | Input: %s",
                model_name.upper(),
                self.infile,
                sys.exc_info()[0],
                text,
                exc_info=True,
            )
            return None

    def _apply_impresso_langident_pipeline(
        self, text: str, model
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Apply impresso_pipelines language identification."""
        try:
            predictions = model(text, diagnostics=True)["diagnostics"]["languages"]
            result = [
                {"lang": r["language"], "prob": prob}
                for r in predictions
                if (prob := r["score"]) > 0.05
            ]
            # probabilites are already rounded in the pipeline
            return result
        except Exception:
            log.error(
                "IMPRESSO-LANGIDENT-PIPELINE-ERROR for %s: %s",
                self.infile,
                sys.exc_info()[0],
            )
            return None

    def _apply_lingua(
        self, text: str, model
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Apply lingua language identification."""
        try:
            confidence_values = model.compute_language_confidence_values(text.lower())
            result = [
                {
                    "lang": confidence.language.iso_code_639_1.name.lower(),
                    "prob": round(confidence.value, self.round_ndigits),
                }
                for confidence in confidence_values
                if confidence.value > 0.05  # Filter out very low confidence predictions
            ]
            return result
        except Exception:
            log.error("LINGUA-ERROR for %s: %s", self.infile, sys.exc_info()[0])
            return None

    def _apply_ocrqa_all_languages(self, text: str, model) -> Optional[dict]:
        """Apply OCR quality assessment for all supported languages.

        :param str text: Text to assess
        :param model: OCR QA pipeline model
        :return: OCR QA results dictionary with language scores or None if failed
        :rtype: Optional[dict]
        """
        try:
            # Get the list of supported languages from the model
            supported_languages = model.SUPPORTED_LANGUAGES

            # Initialize results dictionary - just the scores
            ocrqa_results = {}

            # Run OCR QA for each supported language
            for language in supported_languages:
                try:
                    result = model(text, language=language, model_id=True)
                    # Store the result for this language
                    ocrqa_results[language] = result
                    log.debug(
                        "OCR QA completed for language %s on content item %s",
                        language,
                        getattr(self, "_current_item_id", "unknown"),
                    )
                except Exception as e:
                    log.warning("OCR QA failed for language %s: %s", language, e)
                    ocrqa_results[language] = None

            return ocrqa_results

        except Exception as e:
            log.error("OCR-QA-ALL-LANGUAGES-ERROR for %s: %s", self.infile, e)
            return None

    def _create_base_info(self, content_item: dict) -> dict:
        """Create base information dictionary for a content item."""
        base_info = {
            "tp": content_item["tp"],
            "id": content_item["id"],
            "len": len(content_item.get("ft", "")),
            "orig_lg": content_item.get("lg"),
            "ts": self.ts,
            "langident_systems_version": self.git_describe or __version__,
        }

        # Include text content if debug mode is enabled
        if self.debug:
            base_info["ft"] = content_item.get("ft", "")

        return base_info

    def _is_text_valid_for_lid(self, content_item: dict) -> tuple[bool, str, float]:
        """
        Check if text is valid for language identification.

        Returns:
            Tuple of (is_valid, text, alphabetical_ratio_value)
        """
        if "ft" not in content_item or not isinstance(content_item["ft"], str):
            return False, "", 0.0

        text = content_item["ft"].strip()
        if len(text) < self.minimal_text_length:
            return False, text, 0.0

        alpha_ratio = round(alphabetical_ratio(text), 2)
        if alpha_ratio < self.alphabetical_ratio_threshold:
            return False, text, alpha_ratio

        return True, text, alpha_ratio

    def _get_ocrqa_max_languages(self, ocrqa_result: dict) -> Optional[str]:
        """Get the language(s) with the highest OCR QA score from results.

        :param dict ocrqa_result: OCR QA results dictionary
        :return: String of max language(s), joined by "_" if tied, or None if no valid scores
        :rtype: Optional[str]
        """
        if not ocrqa_result:
            return None

        # Extract valid scores (non-None values that have numeric scores)
        valid_scores = {}
        for lang, result in ocrqa_result.items():
            if result is not None:
                # Try to extract score from result - this depends on OCR QA pipeline output format
                # Assuming the result has a 'score' field, adjust as needed based on actual format
                try:
                    if isinstance(result, dict) and "score" in result:
                        score = result["score"]
                    elif isinstance(result, (int, float)):
                        score = result
                    else:
                        # Try to find a numeric value in the result
                        score = None
                        if isinstance(result, dict):
                            for key in ["score", "quality", "confidence", "value"]:
                                if key in result and isinstance(
                                    result[key], (int, float)
                                ):
                                    score = result[key]
                                    break

                    if score is not None and isinstance(score, (int, float)):
                        valid_scores[lang] = score
                except Exception as e:
                    log.debug(
                        "Could not extract score from OCR QA result for %s: %s", lang, e
                    )
                    continue

        if not valid_scores:
            return None

        # Find maximum score
        max_score = max(valid_scores.values())

        # Get all languages with maximum score
        max_languages = [
            lang for lang, score in valid_scores.items() if score == max_score
        ]

        # Return sorted and joined languages
        return "_".join(sorted(max_languages))

    def _perform_language_identification(
        self, text: str, models: dict, jinfo: dict
    ) -> None:
        """Perform language identification with all configured models."""

        # Store current item ID for OCR QA logging
        self._current_item_id = jinfo["id"]

        # Define model handlers
        model_handlers = {
            "langdetect": lambda: self._apply_langdetect(text),
            "langid": lambda: (
                self._apply_langid(text, models["langid"])
                if "langid" in models
                else None
            ),
            "impresso_ft": lambda: (
                self._apply_fasttext(text, models["impresso_ft"], "impresso_ft")
                if "impresso_ft" in models
                else None
            ),
            "wp_ft": lambda: (
                self._apply_fasttext(text, models["wp_ft"], "wp_ft")
                if "wp_ft" in models
                else None
            ),
            "impresso_langident_pipeline": lambda: (
                self._apply_impresso_langident_pipeline(
                    text, models["impresso_langident_pipeline"]
                )
                if "impresso_langident_pipeline" in models
                else None
            ),
            "lingua": lambda: (
                self._apply_lingua(text, models["lingua"])
                if "lingua" in models
                else None
            ),
        }

        # Apply each requested LID system
        for lid_system in self.lids:
            if lid_system in model_handlers:
                result = model_handlers[lid_system]()
                jinfo[lid_system] = result
                if result is None:
                    log.debug(
                        "No result from %s language identifier for %s",
                        lid_system,
                        self.infile,
                    )
            else:
                log.warning(
                    "No handler defined for LID system %s when processing %s",
                    lid_system,
                    self.infile,
                )
                jinfo[lid_system] = None

        # Apply OCR QA if enabled
        if self.ocrqa and "ocrqa" in models:
            ocrqa_result = self._apply_ocrqa_all_languages(text, models["ocrqa"])
            if ocrqa_result:
                jinfo["ocrqa"] = ocrqa_result
                num_languages = len(ocrqa_result)
                num_successful = len(
                    [score for score in ocrqa_result.values() if score is not None]
                )
                log.debug(
                    "OCR QA completed for %s: %d/%d languages successful",
                    jinfo["id"],
                    num_successful,
                    num_languages,
                )

                # Update OCR QA statistics
                self.stats["ocrqa_processed"] += 1

                # Track languages with maximum OCR QA scores
                max_langs = self._get_ocrqa_max_languages(ocrqa_result)
                if max_langs:
                    if max_langs not in self.stats["ocrqa_max_languages"]:
                        self.stats["ocrqa_max_languages"][max_langs] = 0
                    self.stats["ocrqa_max_languages"][max_langs] += 1

            else:
                jinfo["ocrqa"] = None
                log.debug("No OCR QA result for %s", jinfo["id"])
                self.stats["ocrqa_failed"] += 1

        # Clean up temporary variable
        if hasattr(self, "_current_item_id"):
            delattr(self, "_current_item_id")

    def _check_language_disagreements(self, jinfo: dict) -> None:
        """Check for disagreements between language identifiers and log them."""
        # Extract best predictions from each model that returned results
        best_predictions = {}

        for lid_system in self.lids:
            if lid_system in jinfo and jinfo[lid_system] is not None:
                results = jinfo[lid_system]
                if isinstance(results, list) and len(results) > 0:
                    # Get the top prediction (highest probability)
                    best_lang = results[0]["lang"]
                    best_predictions[lid_system] = best_lang

        # Check if we have at least 2 predictions to compare
        if len(best_predictions) < 2:
            return

        # Check if all predictions agree
        unique_predictions = set(best_predictions.values())
        if len(unique_predictions) > 1:
            # Log disagreement with document ID and all predictions
            predictions_str = ", ".join(
                [f"{lid}:{lang}" for lid, lang in best_predictions.items()]
            )
            log.info("LANGUAGE-DISAGREEMENT %s: %s", jinfo["id"], predictions_str)
            self.stats["language_disagreements"] += 1

            # Create confusion counter key from sorted unique predicted languages
            sorted_languages = sorted(unique_predictions)
            confusion_key = f"LID_DISAGREEMENT_{'_'.join(sorted_languages)}"
            if confusion_key not in self.stats:
                self.stats[confusion_key] = 0
            self.stats[confusion_key] += 1

    def _log_statistics(self):
        """Log processing statistics."""
        total = self.stats["processed_items"]
        if total > 0:
            log.info("STATS-PROCESSED-ITEMS\t%d (100.0%%)", total)
            log.info(
                "STATS-SKIPPED-NO-TEXT\t%d (%.1f%%)",
                self.stats["skipped_no_text"],
                (self.stats["skipped_no_text"] / total) * 100,
            )
            log.info(
                "STATS-SKIPPED-SHORT-TEXT\t%d (%.1f%%)",
                self.stats["skipped_short_text"],
                (self.stats["skipped_short_text"] / total) * 100,
            )
            log.info(
                "STATS-SKIPPED-LOW-ALPHA\t%d (%.1f%%)",
                self.stats["skipped_low_alpha"],
                (self.stats["skipped_low_alpha"] / total) * 100,
            )
            log.info(
                "STATS-LANGUAGE-IDENTIFIED\t%d (%.1f%%)",
                self.stats["language_identified"],
                (self.stats["language_identified"] / total) * 100,
            )
            log.info(
                "STATS-LANGUAGE-DISAGREEMENTS\t%d (%.1f%%)",
                self.stats["language_disagreements"],
                (self.stats["language_disagreements"] / total) * 100,
            )

            # Log confusion counters
            confusion_stats = {
                k: v for k, v in self.stats.items() if k.startswith("LID_DISAGREEMENT_")
            }
            for confusion_key in sorted(confusion_stats.keys()):
                count = confusion_stats[confusion_key]
                log.info(
                    "STATS-%s\t%d (%.1f%%)", confusion_key, count, (count / total) * 100
                )

            # Log OCR QA statistics if enabled
            if self.ocrqa:
                ocrqa_total = self.stats.get("ocrqa_processed", 0) + self.stats.get(
                    "ocrqa_failed", 0
                )
                if ocrqa_total > 0:
                    log.info(
                        "STATS-OCRQA-PROCESSED\t%d (%.1f%%)",
                        self.stats.get("ocrqa_processed", 0),
                        (self.stats.get("ocrqa_processed", 0) / ocrqa_total) * 100,
                    )
                    log.info(
                        "STATS-OCRQA-FAILED\t%d (%.1f%%)",
                        self.stats.get("ocrqa_failed", 0),
                        (self.stats.get("ocrqa_failed", 0) / ocrqa_total) * 100,
                    )

                    # Log top languages with maximum OCR QA scores
                    max_lang_stats = self.stats.get("ocrqa_max_languages", {})
                    if max_lang_stats:
                        log.info("STATS-OCRQA-MAX-LANGUAGES-TOP10:")
                        # Sort by count and show top 10
                        sorted_max_langs = sorted(
                            max_lang_stats.items(), key=lambda x: x[1], reverse=True
                        )[:10]
                        for max_langs, count in sorted_max_langs:
                            percentage = (
                                count / self.stats.get("ocrqa_processed", 1)
                            ) * 100
                            log.info(
                                "STATS-OCRQA-MAX-LANG-%s\t%d (%.1f%%)",
                                max_langs,
                                count,
                                percentage,
                            )
                else:
                    log.info(
                        "STATS-OCRQA-PROCESSED\t%d",
                        self.stats.get("ocrqa_processed", 0),
                    )
                    log.info(
                        "STATS-OCRQA-FAILED\t%d", self.stats.get("ocrqa_failed", 0)
                    )
        else:
            log.info("STATS-PROCESSED-ITEMS\t%d", total)
            log.info("STATS-SKIPPED-NO-TEXT\t%d", self.stats["skipped_no_text"])
            log.info("STATS-SKIPPED-SHORT-TEXT\t%d", self.stats["skipped_short_text"])
            log.info("STATS-SKIPPED-LOW-ALPHA\t%d", self.stats["skipped_low_alpha"])
            log.info("STATS-LANGUAGE-IDENTIFIED\t%d", self.stats["language_identified"])
            log.info(
                "STATS-LANGUAGE-DISAGREEMENTS\t%d", self.stats["language_disagreements"]
            )

            # Log confusion counters
            confusion_stats = {
                k: v for k, v in self.stats.items() if k.startswith("LID_DISAGREEMENT_")
            }
            for confusion_key in sorted(confusion_stats.keys()):
                count = confusion_stats[confusion_key]
                log.info("STATS-%s\t%d", confusion_key, count)

            # Log OCR QA statistics if enabled
            if self.ocrqa:
                log.info(
                    "STATS-OCRQA-PROCESSED\t%d", self.stats.get("ocrqa_processed", 0)
                )
                log.info("STATS-OCRQA-FAILED\t%d", self.stats.get("ocrqa_failed", 0))

                # Log languages with maximum OCR QA scores
                max_lang_stats = self.stats.get("ocrqa_max_languages", {})
                if max_lang_stats:
                    log.info("STATS-OCRQA-MAX-LANGUAGES-TOP10:")
                    sorted_max_langs = sorted(
                        max_lang_stats.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                    for max_langs, count in sorted_max_langs:
                        log.info("STATS-OCRQA-MAX-LANG-%s\t%d", max_langs, count)

    def language_identification(self) -> None:
        """Run multiple language identifications with the models provided and update results."""
        models = self._initialize_models()

        for content_item in self.next_contentitem():
            log.debug("WORKING ON %s", content_item["id"])

            try:
                self.stats["processed_items"] += 1
                jinfo = self._create_base_info(content_item)
                is_valid, text, alpha_ratio = self._is_text_valid_for_lid(content_item)

                if not is_valid:
                    if "ft" not in content_item or not isinstance(
                        content_item["ft"], str
                    ):
                        log.info(
                            "Skipping %s from %s - no valid text field",
                            content_item["id"],
                            self.infile,
                        )
                        self.stats["skipped_no_text"] += 1
                    elif len(text) < self.minimal_text_length:
                        log.info(
                            "Skipping %s from %s - insufficient text length: %d < %d",
                            content_item["id"],
                            self.infile,
                            len(text),
                            self.minimal_text_length,
                        )
                        self.stats["skipped_short_text"] += 1
                    else:
                        log.info(
                            "Skipping %s from %s - low alphabetical ratio: %.2f < %.2f",
                            content_item["id"],
                            self.infile,
                            alpha_ratio,
                            self.alphabetical_ratio_threshold,
                        )
                        self.stats["skipped_low_alpha"] += 1

                    self.results.append(jinfo)
                    continue

                # Text is valid for language identification
                jinfo["alphabetical_ratio"] = round(alpha_ratio, self.round_ndigits)
                self._perform_language_identification(text, models, jinfo)

                # Check for disagreements between language identifiers
                self._check_language_disagreements(jinfo)

                self.stats["language_identified"] += 1
                self.results.append(jinfo)

            except Exception:
                log.error(
                    "PROBLEM processing %s from %s: %s %s %s",
                    content_item.get("id", "unknown"),
                    self.infile,
                    sys.exc_info(),
                    jinfo,
                    content_item,
                )
                exit(1)

    def write_output(self) -> None:
        """Write results to JSON Lines output file."""
        log.info(
            "Writing %d results from %s to %s",
            len(self.results),
            self.infile,
            self.outfile,
        )
        with smart_open.open(self.outfile, mode="w", encoding="utf-8") as f_out:
            for r in self.results:
                f_out.write(
                    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        log.info("Successfully wrote output for %s to %s", self.infile, self.outfile)

    def _extract_text_from_page(self, page_data: dict) -> Dict[str, str]:
        """Extract text from a canonical page format, grouped by content item ID.

        Handles dehyphenation by using 'nf' (normalized form) for hyphenated tokens
        and skipping the first part of hyphenated words marked with 'hy'.

        :param dict page_data: Page data in canonical format
        :return: Dictionary mapping content item IDs to their text content
        :rtype: Dict[str, str]
        """
        content_items = {}

        if "r" not in page_data:
            return content_items

        for region in page_data["r"]:
            content_item_id = region.get("pOf")
            if not content_item_id:
                continue

            region_text = []

            if "p" in region:
                for paragraph in region["p"]:
                    paragraph_text = []

                    if "l" in paragraph:
                        for line in paragraph["l"]:
                            line_text = []

                            if "t" in line:
                                for n, token in enumerate(line["t"]):
                                    if "tx" not in token:
                                        continue

                                    token_text = None
                                    add_token = True

                                    # Handle hyphenation
                                    if "hy" in token and token.get("hy"):
                                        # This is the first part of a hyphenated word, skip it
                                        add_token = False
                                    elif "nf" in token:
                                        # This is the second part with normalized form
                                        token_text = (
                                            token["nf"]
                                            if token["nf"] is not None
                                            else ""
                                        )
                                    else:
                                        # Regular token
                                        token_text = (
                                            token["tx"]
                                            if token["tx"] is not None
                                            else ""
                                        )

                                    # Add the token if it should be included
                                    if add_token and token_text is not None:
                                        line_text.append(token_text)

                                        # Handle spacing: if gn (glue next) is not True, add space
                                        if not token.get("gn", False):
                                            line_text.append(" ")

                            paragraph_text.append("".join(line_text).rstrip())

                    if paragraph_text:
                        region_text.append(" ".join(paragraph_text))

            if region_text:
                region_full_text = " ".join(region_text)
                if content_item_id in content_items:
                    content_items[content_item_id] += " " + region_full_text
                else:
                    content_items[content_item_id] = region_full_text

        return content_items

    def _load_content_item_metadata(self) -> Dict[str, dict]:
        """Load content item metadata from issue file.

        :return: Dictionary mapping content item IDs to their metadata
        :rtype: Dict[str, dict]
        """
        content_item_metadata = {}

        if not self.issue_file:
            return content_item_metadata

        if self.issue_file.startswith("s3://"):
            transport_params = {"client": self.s3_client}
        else:
            transport_params = {}

        log.info("Loading content item metadata from issue file: %s", self.issue_file)

        with smart_open.open(
            self.issue_file, transport_params=transport_params, encoding="utf-8"
        ) as reader:
            for line in reader:
                if not line.strip():
                    continue

                issue_data = json.loads(line)

                # Extract content items from issue
                if "i" in issue_data:
                    for content_item in issue_data["i"]:
                        if "m" in content_item:
                            metadata = content_item["m"]
                            content_item_id = metadata.get("id")
                            if content_item_id:
                                content_item_metadata[content_item_id] = {
                                    "orig_lg": metadata.get("l"),
                                    "tp": metadata.get("tp", "article"),
                                    "title": metadata.get("t"),
                                    "pages": metadata.get("pp", []),
                                }

        log.info(
            "Loaded metadata for %d content items from %s",
            len(content_item_metadata),
            self.issue_file,
        )

        return content_item_metadata

    def _build_content_items_from_canonical(self) -> Dict[str, dict]:
        """Build content items dictionary from canonical format pages.

        :return: Dictionary mapping content item IDs to content item objects
        :rtype: Dict[str, dict]
        """
        # First load content item metadata from issue file
        content_item_metadata = self._load_content_item_metadata()

        content_items = {}

        # Handle S3 prefix patterns
        if self.infile.startswith("s3://") and not self.infile.endswith(".jsonl.bz2"):
            # This is a prefix pattern, list all matching files
            # Parse S3 path
            s3_path = self.infile[5:]  # Remove 's3://'
            bucket_name, prefix = s3_path.split("/", 1)

            # Use yield_s3_objects to get all matching files
            page_files = []
            for key in yield_s3_objects(bucket_name, prefix):
                if key.endswith(".jsonl.bz2"):
                    page_files.append(f"s3://{bucket_name}/{key}")

            log.info(
                "Found %d files matching prefix %s",
                len(page_files),
                self.infile,
            )

            if not page_files:
                log.warning("No files found matching S3 prefix: %s", self.infile)
                return content_items
        else:
            # Single file or local file
            page_files = [self.infile]

        # Process each page file
        for page_file in page_files:
            log.info("Processing page file: %s", page_file)

            if page_file.startswith("s3://"):
                transport_params = {"client": self.s3_client}
            else:
                transport_params = {}

            try:
                with smart_open.open(
                    page_file, transport_params=transport_params, encoding="utf-8"
                ) as reader:
                    for line in reader:
                        if not line.strip():
                            continue

                        page_data = json.loads(line)
                        page_content_items = self._extract_text_from_page(page_data)

                        for content_item_id, text in page_content_items.items():
                            if content_item_id in content_items:
                                # Append text from this page to existing content item
                                content_items[content_item_id]["ft"] += " " + text
                            else:
                                # Create new content item with metadata from issue file
                                metadata = content_item_metadata.get(
                                    content_item_id, {}
                                )
                                content_items[content_item_id] = {
                                    "id": content_item_id,
                                    "ft": text,
                                    "tp": metadata.get("tp", "article"),
                                    "lg": metadata.get("orig_lg"),
                                }

            except Exception as e:
                log.error("Error processing page file %s: %s", page_file, e)
                continue

        # Clean up text for all content items
        for content_item in content_items.values():
            content_item["ft"] = " ".join(content_item["ft"].split())

        log.info(
            "Extracted %d content items from %d page files with prefix %s",
            len(content_items),
            len(page_files),
            self.infile,
        )

        return content_items

    def next_contentitem(self) -> Iterable[dict]:
        """Yield each content item from the input file."""
        if self.format == "canonical":
            # For canonical format, first build all content items from pages
            content_items = self._build_content_items_from_canonical()
            for content_item in content_items.values():
                yield content_item
        else:
            # Original rebuilt format processing
            if self.infile.startswith("s3://"):
                transport_params = {"client": self.s3_client}
            else:
                transport_params = {}
            with smart_open.open(
                self.infile, transport_params=transport_params, encoding="utf-8"
            ) as reader:
                for line in reader:
                    if line.strip():
                        yield json.loads(line)


def main():
    import argparse

    DESCRIPTION = (
        "Identify languages and their probabilities with different LID systems."
    )

    EPILOG = (
        "All tools use two-letter ISO 639-1 codes, except wp_ft which "
        "recognizes additional languages identifiable only by 3 letter codes."
    )
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG)

    # Input and Output Files
    parser.add_argument(
        "-i",
        "--infile",
        default="/dev/stdin",
        help=(
            "Path to input file. For rebuilt format: single file path. For canonical"
            " format: S3 prefix (e.g.,"
            " s3://bucket/NEWSPAPER/pages/NEWSPAPER-YEAR/NEWSPAPER-) or single file"
            " path (default %(default)s)"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="/dev/stdout",
        help="path to output file for impresso lid json format (default %(default)s)",
    )

    # Format Selection
    parser.add_argument(
        "--format",
        choices=["rebuilt", "canonical"],
        default="rebuilt",
        help=(
            "input format type: 'rebuilt' for traditional format or 'canonical' for"
            " page schema format (default %(default)s)"
        ),
    )

    # Issue file for canonical format
    parser.add_argument(
        "--issue-file",
        help=(
            "Path to issue metadata file (required for canonical format). "
            "Example: s3://bucket/NEWSPAPER/issues/NEWSPAPER-YEAR.issue.jsonl.bz2"
        ),
        metavar="FILE",
    )

    # Language Identification Systems
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
            "names of all LID systems (e.g. langdetect, langid) to use. Do not add"
            " orig_lg here! %(default)s"
        ),
    )

    # Models
    parser.add_argument(
        "--impresso-ft",
        default=None,
        help="binary fasttext LID impresso model labeled impresso_ft in the output)",
        metavar="FILE",
    )
    parser.add_argument(
        "--wp-ft",
        default=None,
        help="binary fasttext wikipedia LID model labeled wp_ft in the output ",
        metavar="FT2",
    )

    # Text Length and Precision
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
        "--round-ndigits",
        default=3,
        type=int,
        help="round floats in the output to n digits (default %(default)s)",
    )

    # Logging and Verbosity
    parser.add_argument(
        "--log-file", dest="log_file", help="Write log to FILE", metavar="FILE"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s)",
    )

    # Version Information
    parser.add_argument(
        "--git-describe",
        type=str,
        default="",
        help=(
            "output of git describe command for ingesting git version into JSON as"
            " version string"
        ),
    )

    # Add alphabetical_ratio_threshold to command-line arguments
    parser.add_argument(
        "--alphabetical-ratio-threshold",
        default=0.0,
        type=float,
        help=(
            "Threshold for alphabetical ratio below which language identification is"
            " skipped (default %(default)s)"
        ),
    )

    # Debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include full text content in output for debugging purposes",
    )

    # Add OCR QA option
    parser.add_argument(
        "--ocrqa",
        action="store_true",
        help=(
            "Enable OCR quality assessment using impresso_pipelines.ocrqa for all"
            " supported languages"
        ),
    )

    arguments = parser.parse_args()

    # Validate format-specific requirements
    if arguments.format == "canonical" and not arguments.issue_file:
        parser.error("--issue-file is required when using --format=canonical")

    setup_logging(arguments.log_level, arguments.log_file)

    log.info("%s", arguments)

    # Directly call LanguageIdentifier with relevant arguments
    processor = LanguageIdentifier(
        infile=arguments.infile,
        outfile=arguments.outfile,
        impresso_ft=arguments.impresso_ft,
        wp_ft=arguments.wp_ft,
        minimal_text_length=arguments.minimal_text_length,
        lids=arguments.lids,
        round_ndigits=arguments.round_ndigits,
        git_describe=arguments.git_describe,
        alphabetical_ratio_threshold=arguments.alphabetical_ratio_threshold,
        format=arguments.format,
        debug=arguments.debug,
        issue_file=arguments.issue_file,
        ocrqa=arguments.ocrqa,
    )
    processor.run()


if __name__ == "__main__":
    main()
    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    setup_logging(log_levels[arguments.verbose], arguments.logfile)

    log.info("%s", arguments)

    # Directly call LanguageIdentifier with relevant arguments
    processor = LanguageIdentifier(
        infile=arguments.infile,
        outfile=arguments.outfile,
        impresso_ft=arguments.impresso_ft,
        wp_ft=arguments.wp_ft,
        minimal_text_length=arguments.minimal_text_length,
        lids=arguments.lids,
        round_ndigits=arguments.round_ndigits,
        git_describe=arguments.git_describe,
        alphabetical_ratio_threshold=arguments.alphabetical_ratio_threshold,
        format=arguments.format,
        debug=arguments.debug,
        issue_file=arguments.issue_file,
        ocrqa=arguments.ocrqa,
    )
    processor.run()


if __name__ == "__main__":
    main()
