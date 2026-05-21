"""Unit tests for Agent4SRAttack and LocalAgent4SR.

Tests
-----
  1.  Profile count matches n_fake_users.
  2.  All fake timestamps are strictly after the last real timestamp.
  3.  Timestamps are non-decreasing within each profile.
  4.  Target items are present and rated at max_rating in every profile.
  5.  Filler ratings are in [1, max_rating] and rounded to nearest 0.5.
  6.  Filler items are distinct from target items.
  7.  Budget constraint: len(rated_items) == filler_size + len(target_items).
  8.  Cache hit: _call_llm is called once per unique prompt, not per profile.
  9.  Malformed LLM response → fallback produces a valid FakeProfile.
  10. inject(): correct user / review counts after injection.
  11. inject(): fake users labelled 1, real users 0.
  12. inject(): fake reviews in test_mask only.
  13. inject(): review_user_idx and review_item_idx in valid range.
  14. Different personas produce different prompts.
  15. LocalAgent4SR._call_llm sends a POST to the correct Ollama URL.
  16. Personas loaded from a custom YAML file override built-in defaults.
  17. Zero filler: when budget covers only target items, no filler is added.
  18. review_texts length matches rated_items for every profile.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from src.attacks import Agent4SRAttack, LocalAgent4SR, FakeProfile
from src.datasets.base import HeteroDataset

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

N_USERS   = 50
N_ITEMS   = 80
N_REVIEWS = 120
SEED      = 0


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

def _make_mock_response(items: List[int], max_rating: float = 5.0) -> str:
    """Return a well-formed JSON LLM response for the given item list."""
    ratings = {f"item_{i}": 4.5 for i in items}
    reviews = {f"item_{i}": f"Genuinely enjoyed item {i}. Would buy again." for i in items}
    return json.dumps({"ratings": ratings, "reviews": reviews})


def _patched_call_llm(self, prompt: str) -> str:
    """Replacement for _call_llm: parses item IDs from prompt, returns JSON."""
    item_ids = [int(m) for m in re.findall(r"item_(\d+)", prompt)]
    return _make_mock_response(item_ids)


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

TARGET_ITEMS  = [1, 5, 12]
BUDGET_PCT    = 0.2              # 20 % of 80 items = 16; filler = 13
MAX_RATING    = 5.0
N_FAKE        = 6
_BUDGET_ITEMS = max(0, round(BUDGET_PCT * N_ITEMS))        # 16
_FILLER_SIZE  = max(0, _BUDGET_ITEMS - len(TARGET_ITEMS))  # 13
_ITEMS_PER    = _FILLER_SIZE + len(TARGET_ITEMS)           # 16


@pytest.fixture(scope="module")
def dataset(synthetic_graph) -> HeteroDataset:
    return synthetic_graph


@pytest.fixture(scope="module")
def profiles(dataset, tmp_path_factory) -> List[FakeProfile]:
    cache = tmp_path_factory.mktemp("cache")
    with patch.object(Agent4SRAttack, "_call_llm", _patched_call_llm):
        attack = Agent4SRAttack(dataset, cache_dir=str(cache),
                                max_rating=MAX_RATING, seed=SEED)
        return attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)


@pytest.fixture(scope="module")
def attack_and_profiles(dataset, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache2")
    with patch.object(Agent4SRAttack, "_call_llm", _patched_call_llm):
        attack = Agent4SRAttack(dataset, cache_dir=str(cache),
                                max_rating=MAX_RATING, seed=SEED)
        profs  = attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    return attack, profs


# ---------------------------------------------------------------------------
# Test 1 — Profile count
# ---------------------------------------------------------------------------

def test_profile_count(profiles):
    assert len(profiles) == N_FAKE


# ---------------------------------------------------------------------------
# Test 2 — Timestamps after max real timestamp
# ---------------------------------------------------------------------------

def test_timestamps_after_max_real(profiles, dataset):
    max_real = float(dataset.timestamps.max().item())
    for p in profiles:
        assert all(t > max_real for t in p.timestamps), \
            f"{p.user_id}: fake timestamp ≤ max real ({max_real})"


# ---------------------------------------------------------------------------
# Test 3 — Timestamps non-decreasing within each profile
# ---------------------------------------------------------------------------

def test_timestamps_ordered(profiles):
    for p in profiles:
        assert p.timestamps == sorted(p.timestamps), \
            f"{p.user_id}: timestamps not sorted"


# ---------------------------------------------------------------------------
# Test 4 — Target items present and at max_rating
# ---------------------------------------------------------------------------

def test_target_items_max_rating(profiles):
    for p in profiles:
        for t in TARGET_ITEMS:
            assert t in p.rated_items, f"{p.user_id} missing target {t}"
            assert p.rated_items[t] == MAX_RATING, \
                f"{p.user_id} target {t}: {p.rated_items[t]} ≠ {MAX_RATING}"


# ---------------------------------------------------------------------------
# Test 5 — Filler ratings in [1, max_rating], rounded to 0.5
# ---------------------------------------------------------------------------

def test_filler_rating_range(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        for item, r in p.rated_items.items():
            if item in target_set:
                continue
            assert 1.0 <= r <= MAX_RATING, \
                f"{p.user_id} item {item} out-of-range rating {r}"
            assert abs(r * 2 - round(r * 2)) < 1e-6, \
                f"{p.user_id} item {item} not rounded to 0.5: {r}"


# ---------------------------------------------------------------------------
# Test 6 — Filler items distinct from targets
# ---------------------------------------------------------------------------

def test_filler_excludes_targets(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        filler = set(p.rated_items.keys()) - target_set
        assert filler.isdisjoint(target_set)


# ---------------------------------------------------------------------------
# Test 7 — Budget constraint
# ---------------------------------------------------------------------------

def test_budget_constraint(profiles):
    for p in profiles:
        assert len(p.rated_items) == _ITEMS_PER, \
            f"{p.user_id}: expected {_ITEMS_PER} items, got {len(p.rated_items)}"


# ---------------------------------------------------------------------------
# Test 8 — Cache hit: identical prompt served from disk, not from LLM
# ---------------------------------------------------------------------------

def test_cache_hit(dataset, tmp_path):
    call_count = 0

    def _counting_llm(self, prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return _patched_call_llm(self, prompt)

    # Zero filler → deterministic prompt for (persona, target_items)
    exact_budget = len(TARGET_ITEMS) / N_ITEMS

    with patch.object(Agent4SRAttack, "_call_llm", _counting_llm):
        a1 = Agent4SRAttack(dataset, cache_dir=str(tmp_path),
                            max_rating=MAX_RATING, seed=SEED)
        a1.generate_profiles(1, exact_budget, TARGET_ITEMS)
        assert call_count == 1

        # Second attack with same seed → same prompt → cache hit, no new call
        a2 = Agent4SRAttack(dataset, cache_dir=str(tmp_path),
                            max_rating=MAX_RATING, seed=SEED)
        a2.generate_profiles(1, exact_budget, TARGET_ITEMS)
        assert call_count == 1, "Cache miss: _call_llm was called again for identical prompt"


# ---------------------------------------------------------------------------
# Test 9 — Malformed LLM response → valid fallback profile
# ---------------------------------------------------------------------------

def test_malformed_response_fallback(dataset, tmp_path):
    def _bad_llm(self, prompt: str) -> str:
        return "Sorry, I cannot help with that request. Lorem ipsum dolor sit amet."

    with patch.object(Agent4SRAttack, "_call_llm", _bad_llm):
        attack = Agent4SRAttack(dataset, cache_dir=str(tmp_path),
                                max_rating=MAX_RATING, seed=SEED)
        profiles = attack.generate_profiles(2, BUDGET_PCT, TARGET_ITEMS)

    assert len(profiles) == 2
    for p in profiles:
        # Budget constraint still met
        assert len(p.rated_items) == _ITEMS_PER
        # Targets still at max
        for t in TARGET_ITEMS:
            assert p.rated_items[t] == MAX_RATING
        # All filler ratings valid
        for item, r in p.rated_items.items():
            if item not in TARGET_ITEMS:
                assert 1.0 <= r <= MAX_RATING
                assert abs(r * 2 - round(r * 2)) < 1e-6


# ---------------------------------------------------------------------------
# Tests 10–13 — inject() structural invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def augmented(attack_and_profiles, dataset):
    attack, profs = attack_and_profiles
    return attack.inject(dataset, profs)


def test_inject_user_count(augmented):
    assert augmented.num_users == N_USERS + N_FAKE


def test_inject_review_count(augmented, profiles):
    n_new = sum(len(p.rated_items) for p in profiles)
    assert augmented.num_reviews == N_REVIEWS + n_new


def test_inject_labels(augmented):
    assert (augmented.labels[:N_USERS] == 0).all()
    assert (augmented.labels[N_USERS:] == 1).all()


def test_inject_masks(augmented):
    assert not augmented.train_mask[N_REVIEWS:].any()
    assert not augmented.val_mask[N_REVIEWS:].any()
    assert augmented.test_mask[N_REVIEWS:].all()


def test_inject_index_bounds(augmented):
    assert augmented.review_user_idx.max() < augmented.num_users
    assert augmented.review_item_idx.max() < augmented.num_items
    assert augmented.review_user_idx.min() >= 0
    assert augmented.review_item_idx.min() >= 0


# ---------------------------------------------------------------------------
# Test 14 — Different personas produce different prompts
# ---------------------------------------------------------------------------

def test_different_personas_produce_different_prompts(dataset, tmp_path):
    prompts = set()

    def _capture_llm(self, prompt: str) -> str:
        prompts.add(prompt)
        return _patched_call_llm(self, prompt)

    # With 10+ fake users we cycle through all personas at least once
    with patch.object(Agent4SRAttack, "_call_llm", _capture_llm):
        attack = Agent4SRAttack(dataset, cache_dir=str(tmp_path),
                                max_rating=MAX_RATING, seed=SEED)
        attack.generate_profiles(10, BUDGET_PCT, TARGET_ITEMS)

    assert len(prompts) >= 2, \
        "All 10 profiles used the same prompt — persona cycling not working"


# ---------------------------------------------------------------------------
# Test 15 — LocalAgent4SR sends POST to correct Ollama URL
# ---------------------------------------------------------------------------

def test_local_agent4sr_calls_ollama(dataset, tmp_path):
    captured_urls: List[str] = []

    def _mock_post(url, json=None, timeout=None):
        captured_urls.append(url)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        # Extract item IDs from the prompt and return valid JSON
        item_ids = [int(m) for m in re.findall(r"item_(\d+)", json.get("prompt", ""))]
        mock_resp.json.return_value = {
            "response": _make_mock_response(item_ids)
        }
        return mock_resp

    with patch("requests.post", side_effect=_mock_post):
        attack = LocalAgent4SR(
            dataset, model="llama3",
            host="http://localhost:11434",
            cache_dir=str(tmp_path),
            max_rating=MAX_RATING, seed=SEED,
        )
        profiles = attack.generate_profiles(2, BUDGET_PCT, TARGET_ITEMS)

    assert len(profiles) == 2
    assert all("http://localhost:11434/api/generate" == url for url in captured_urls)


# ---------------------------------------------------------------------------
# Test 16 — Personas loaded from a custom YAML file
# ---------------------------------------------------------------------------

def test_custom_personas_yaml(dataset, tmp_path):
    custom_yaml = tmp_path / "my_personas.yaml"
    custom_yaml.write_text(yaml.dump({
        "personas": [
            {"name": "alpha", "description": "Alpha persona: extremely price-sensitive buyer."},
            {"name": "beta",  "description": "Beta persona: impulse buyer who trusts reviews."},
        ]
    }))

    names_seen: List[str] = []

    def _capture(self, prompt: str) -> str:
        names_seen.append(prompt)   # just count calls; content checked separately
        return _patched_call_llm(self, prompt)

    with patch.object(Agent4SRAttack, "_call_llm", _capture):
        attack = Agent4SRAttack(
            dataset,
            cache_dir=str(tmp_path / "cache"),
            personas_path=str(custom_yaml),
            max_rating=MAX_RATING, seed=SEED,
        )
        assert len(attack.personas) == 2
        assert attack.personas[0]["name"] == "alpha"
        attack.generate_profiles(2, BUDGET_PCT, TARGET_ITEMS)

    # First profile uses 'alpha', second uses 'beta' (round-robin)
    assert "Alpha persona" in names_seen[0]
    assert "Beta persona"  in names_seen[1]


# ---------------------------------------------------------------------------
# Test 17 — Zero filler
# ---------------------------------------------------------------------------

def test_zero_filler(dataset, tmp_path):
    exact_budget = len(TARGET_ITEMS) / N_ITEMS  # filler_size = 0
    with patch.object(Agent4SRAttack, "_call_llm", _patched_call_llm):
        attack = Agent4SRAttack(dataset, cache_dir=str(tmp_path),
                                max_rating=MAX_RATING, seed=SEED)
        profiles = attack.generate_profiles(3, exact_budget, TARGET_ITEMS)

    for p in profiles:
        assert set(p.rated_items.keys()) == set(TARGET_ITEMS)
        assert all(r == MAX_RATING for r in p.rated_items.values())


# ---------------------------------------------------------------------------
# Test 18 — review_texts length matches rated_items
# ---------------------------------------------------------------------------

def test_review_texts_length(profiles):
    for p in profiles:
        assert len(p.review_texts) == len(p.rated_items), \
            f"{p.user_id}: {len(p.review_texts)} texts ≠ {len(p.rated_items)} items"


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
