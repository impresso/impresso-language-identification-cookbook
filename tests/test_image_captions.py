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
