"""Tests for language identification of image captions."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from impresso_ensemble_lid import ImpressoLanguageIdentifierEnsemble
from impresso_langident_systems import ImpressoLanguageIdentifierSystems
from newspaper_statistics import AggregatorLID


def _metadata(tp, title):
    return {"tp": tp, "title": title, "orig_lg": "de"}


def test_captioned_image_is_added_with_caption_as_text():
    items = {}
    ImpressoLanguageIdentifierSystems._add_image_captions(
        items, {"x-i0001": _metadata("image", "Une robe de soie")}
    )
    assert items == {
        "x-i0001": {
            "id": "x-i0001",
            "ft": "Une robe de soie",
            "tp": "image",
            "lg": "de",
        }
    }


def test_caption_replaces_text_extracted_from_pages():
    items = {"x-i0001": {"id": "x-i0001", "ft": "page text", "tp": "image", "lg": "de"}}
    ImpressoLanguageIdentifierSystems._add_image_captions(
        items, {"x-i0001": _metadata("image", "caption")}
    )
    assert items["x-i0001"]["ft"] == "caption"


def test_uncaptioned_images_and_other_types_are_not_added():
    items = {}
    ImpressoLanguageIdentifierSystems._add_image_captions(
        items,
        {
            "x-i0001": _metadata("image", None),
            "x-i0002": _metadata("image", "   "),
            "x-i0003": _metadata("article", "A title"),
        },
    )
    assert items == {}


def _ensemble():
    ensemble = ImpressoLanguageIdentifierEnsemble.__new__(
        ImpressoLanguageIdentifierEnsemble
    )
    ensemble.ts = "2026-10-31T00:00:00Z"
    ensemble.git_describe = "test"
    ensemble.threshold_confidence_orig_lg = 0.75
    ensemble.newspaper_stats = {"overall_orig_lg_support": 0.1, "dominant_language": "fr"}
    ensemble.lids = {"langid"}
    ensemble.minimal_text_length = 50
    return ensemble


def _image(length):
    return {
        "id": "x-1900-01-01-a-i0001",
        "tp": "image",
        "len": length,
        "orig_lg": "de",
        "alphabetical_ratio": 0.8,
    }


def test_short_caption_follows_normal_rules():
    decided = _ensemble().decide_lg(_image(10))
    assert decided["lg"] == "fr"
    assert decided["lg_decision"] == "dominant-by-lowvote"


def test_image_without_text_gets_no_language():
    assert "lg" not in _ensemble().decide_lg(_image(0))


def test_captions_are_excluded_from_newspaper_statistics():
    aggregator = AggregatorLID.__new__(AggregatorLID)
    aggregator.newspaper = "x"
    aggregator.contentitem_type_distribution = Counter()
    aggregator.content_length_stats = Counter()
    aggregator.get_next_contentitem = lambda: iter(
        [{"id": "x-1900-01-01-a-i0001", "tp": "image", "len": 100}]
    )
    aggregator.collect_statistics()
    assert aggregator.contentitem_type_distribution == {"image": 1}
    assert not aggregator.content_length_stats


def _trusting_ensemble():
    ensemble = _ensemble()
    ensemble.admissible_languages = None
    ensemble.exclude_lb = set()
    ensemble.alphabetical_ratio_threshold = 0.5
    ensemble.minimal_lid_probability = 0.5
    ensemble.minimal_voting_score = 0.5
    ensemble.weight_lb_impresso_ft = 3.0
    ensemble.dominant_language_threshold = 0.9
    ensemble.infile = "x.jsonl.bz2"
    ensemble.newspaper_stats = {
        "overall_orig_lg_support": 0.98,
        "dominant_language": "fr",
        "lg_support": {
            "orig_lg": {"de": 0.98, "en": 0.4},
            "langid": {"de": 0.9, "fr": 0.9},
            "wp_ft": {"de": 0.9, "fr": 0.9},
        },
    }
    return ensemble


def test_reliable_original_language_decides_without_predictions():
    decided = _trusting_ensemble().decide_lg(_image(10))
    assert decided["lg"] == "de"
    assert decided["lg_decision"] == "voting"
    assert decided["votes"] == [{"lang": "de", "vote": 0.98}]


def test_unreliable_original_language_does_not_decide():
    image = _image(10)
    image["orig_lg"] = "en"
    decided = _trusting_ensemble().decide_lg(image)
    assert decided["lg"] == "fr"
    assert decided["lg_decision"] == "dominant-by-lowvote"


def test_missing_original_language_falls_back_to_normal_rules():
    image = _image(10)
    image["orig_lg"] = None
    assert _trusting_ensemble().decide_lg(image)["lg"] == "fr"


def test_unanimous_systems_overrule_original_language():
    image = _image(200)
    image["langid"] = [{"lang": "fr", "prob": 0.9}]
    decided = _trusting_ensemble().decide_lg(image)
    assert decided["lg"] == "fr"
    assert decided["lg_decision"] == "all"


def test_original_language_breaks_tie_between_systems():
    image = _image(200)
    image["langid"] = [{"lang": "fr", "prob": 0.9}]
    image["wp_ft"] = [{"lang": "de", "prob": 0.9}]
    ensemble = _trusting_ensemble()
    ensemble.lids = {"langid", "wp_ft"}
    decided = ensemble.decide_lg(image)
    assert decided["lg"] == "de"
    assert decided["lg_decision"] == "voting"


def test_articles_do_not_get_an_original_language_vote():
    article = _image(200)
    article["tp"] = "article"
    article["langid"] = [{"lang": "fr", "prob": 0.9}]
    article["wp_ft"] = [{"lang": "de", "prob": 0.8}]
    ensemble = _trusting_ensemble()
    ensemble.lids = {"langid", "wp_ft"}
    assert ensemble.decide_lg(article)["lg"] == "fr"
