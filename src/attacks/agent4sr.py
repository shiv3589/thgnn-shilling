"""Agent4SR — LLM-agent shilling attack.

Uses a language model to generate realistic fake user profiles (ratings +
review texts) that are statistically and linguistically indistinguishable
from genuine users.

Architecture
------------
For each fake user:
  1. Sample a persona from ``configs/agent4sr_personas.yaml``.
  2. Randomly select ``filler_size`` non-target items.
  3. Build a structured prompt embedding dataset context, the persona, and
     the full item list (target items marked as highly relevant).
  4. Call the LLM; cache the raw response keyed by SHA-256(prompt).
  5. Parse the JSON response → ratings dict + review texts list.
  6. Discretise ratings to 0.5-star increments; guarantee targets at max.

Caching
-------
Every LLM response is stored as ``{cache_dir}/{sha256(prompt)}.json``.
Subsequent calls with an identical prompt are served from cache, making
repeated experiments fast and free.

Usage
-----
    # Cloud (requires OPENAI_API_KEY environment variable)
    attack = Agent4SRAttack(dataset, model="gpt-4o",
                            cache_dir="data/agent4sr_cache/")
    profiles = attack.generate_profiles(50, budget_pct=0.1,
                                        target_items=[3, 9, 42])

    # Local (requires `ollama serve` with llama3 pulled)
    attack = LocalAgent4SR(dataset, model="llama3")
    profiles = attack.generate_profiles(10, budget_pct=0.1,
                                        target_items=[3, 9, 42])
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.datasets.base import HeteroDataset
from .base import FakeProfile, ShillingAttack

logger = logging.getLogger(__name__)

# LLM completions can realistically enumerate ratings/reviews for only a
# small number of items; capping prevents truncated, unparseable JSON on
# large-catalog datasets (see paper footnote in experiments.tex).
MAX_ITEMS_PER_PROFILE = 50

# Fallback personas used when the YAML file is missing or unreadable.
_DEFAULT_PERSONAS: List[Dict] = [
    {
        "name": "casual_reader",
        "description": (
            "A casual reader who gravitates towards bestsellers and widely popular titles. "
            "Tends to rate generously — usually 4 or 5 stars — and writes short, enthusiastic "
            "reviews. Drawn to feel-good stories, self-help books, and light fiction."
        ),
    },
    {
        "name": "genre_fiction_fan",
        "description": (
            "A devoted genre fiction fan who loves science fiction, fantasy, and thrillers. "
            "Writes detailed reviews comparing books to others in the genre. Rates critically "
            "but rewards originality and world-building with high scores."
        ),
    },
    {
        "name": "gift_buyer",
        "description": (
            "A practical buyer who mostly purchases items as gifts for friends and family. "
            "Reviews focus on packaging, suitability as a gift, and recipient reactions rather "
            "than personal taste. Ratings are usually positive when the recipient is happy."
        ),
    },
    {
        "name": "academic_reader",
        "description": (
            "An academic who reads primarily non-fiction: history, science, biography, and "
            "current affairs. Reviews are thoughtful and critical, evaluating depth of research "
            "and quality of argument. Rarely gives 5 stars; 3–4 is considered good."
        ),
    },
    {
        "name": "budget_shopper",
        "description": (
            "A price-conscious shopper who carefully compares options before buying. "
            "Reviews emphasise value for money and durability. Rates poorly if the product "
            "feels overpriced, even if quality is acceptable."
        ),
    },
    {
        "name": "early_adopter",
        "description": (
            "An enthusiast who buys new releases and trending items the moment they appear. "
            "Short, punchy reviews written right after purchase. Generous rater who loves "
            "novelty; often gives 5 stars to first impressions."
        ),
    },
    {
        "name": "literary_critic",
        "description": (
            "A discerning reader with high literary standards. Reviews are long and nuanced, "
            "weighing prose quality, character depth, and thematic richness. Ratings below 3 "
            "are common for anything that feels clichéd or poorly edited."
        ),
    },
    {
        "name": "commuter_reader",
        "description": (
            "A busy professional who reads during commutes — prefers audiobooks and short, "
            "engaging works that can be consumed in quick sessions. Reviews mention pacing and "
            "narrator quality. Generous ratings for anything that kept attention on a busy day."
        ),
    },
    {
        "name": "young_adult_fan",
        "description": (
            "A teenager or young adult who loves YA fiction, romance, and coming-of-age stories. "
            "Reviews are informal and energetic, often using phrases like 'I couldn't put it down'. "
            "Rates 5 stars for emotional impact and relatable characters."
        ),
    },
    {
        "name": "lifelong_learner",
        "description": (
            "A curious retiree who reads across genres to keep mentally active. Enjoys memoirs, "
            "travel writing, popular science, and classic literature. Reviews are warm and "
            "conversational, often sharing personal anecdotes the book inspired."
        ),
    },
]


# ---------------------------------------------------------------------------
# Agent4SRAttack
# ---------------------------------------------------------------------------

class Agent4SRAttack(ShillingAttack):
    """LLM-agent shilling attack (cloud backend, default: gpt-4o).

    Parameters
    ----------
    dataset      : HeteroDataset — clean reference dataset
    model        : str  — OpenAI model name (default ``"gpt-4o"``)
    cache_dir    : str  — directory for prompt-response cache (created if absent)
    personas_path: str  — path to a YAML personas file; if None, searches
                          ``configs/agent4sr_personas.yaml`` then falls back
                          to built-in defaults
    max_rating   : float — upper bound of the rating scale (default 5.0)
    seed         : int  — RNG seed
    """

    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(
        self,
        dataset: HeteroDataset,
        model: str = "gpt-4o",
        cache_dir: str = "data/agent4sr_cache/",
        personas_path: Optional[str] = None,
        max_rating: float = 5.0,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, seed)
        self.model      = model
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_rating = max_rating
        self.personas   = self._load_personas(personas_path)

    # ------------------------------------------------------------------
    # Persona loading
    # ------------------------------------------------------------------

    def _load_personas(self, path: Optional[str]) -> List[Dict]:
        """Load personas from YAML; fall back to built-in defaults."""
        if path is not None:
            candidates = [Path(path)]
        else:
            # Default: project-root/configs/agent4sr_personas.yaml
            project_root = Path(__file__).resolve().parent.parent.parent
            candidates   = [project_root / "configs" / "agent4sr_personas.yaml"]

        for p in candidates:
            if p.exists():
                try:
                    with open(p) as f:
                        data = yaml.safe_load(f)
                    personas = data.get("personas", [])
                    if personas:
                        logger.info("Agent4SR: loaded %d personas from %s", len(personas), p)
                        return personas
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Agent4SR: could not load personas from %s: %s", p, exc)

        logger.info("Agent4SR: using %d built-in personas.", len(_DEFAULT_PERSONAS))
        return _DEFAULT_PERSONAS

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        stats: Dict,
        target_items: List[int],
        filler_items: List[int],
        persona: Dict,
    ) -> str:
        """Return a structured prompt asking the LLM to play *persona* and review items."""
        target_set = set(target_items)
        # Fixed ordering: targets first, then filler — deterministic for caching
        all_items = list(target_items) + [i for i in filler_items if i not in target_set]

        item_lines: List[str] = []
        for item in all_items:
            if item in target_set:
                item_lines.append(
                    f"  - item_{item}  "
                    f"[strongly matches your interests — loved it]"
                )
            else:
                item_lines.append(f"  - item_{item}")
        items_str = "\n".join(item_lines)

        valid_values = ", ".join(
            str(v / 2) for v in range(2, int(self.max_rating * 2) + 1)
        )

        prompt = (
            f'You are simulating a real user on a recommendation platform called "{stats["name"]}".\n'
            "\n"
            "USER PERSONA\n"
            f"{persona['description']}\n"
            "\n"
            "PLATFORM CONTEXT\n"
            f"The platform has {stats['num_users']} users and {stats['num_items']} items.\n"
            f"Average user rating: {stats['mean_rating']:.1f} / {self.max_rating:.0f} stars.\n"
            "\n"
            "TASK\n"
            "You have purchased and used each item below. For every item, provide:\n"
            "  1. A star rating (allowed values: " + valid_values + ")\n"
            "  2. A brief, natural review (2–3 sentences) consistent with your persona\n"
            "\n"
            "Items:\n"
            f"{items_str}\n"
            "\n"
            "Items marked '[strongly matches your interests — loved it]' are among your "
            f"favourites — rate them {self.max_rating:.0f} stars.\n"
            "\n"
            "RESPONSE\n"
            "Reply with ONLY a JSON object — no markdown fences, no explanation:\n"
            '{"ratings": {"item_42": 5.0, "item_7": 3.5, ...}, '
            '"reviews": {"item_42": "Great product...", "item_7": "Decent but...", ...}}'
        )
        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        text: str,
        target_items: List[int],
        filler_items: List[int],
        mean_rating: float,
        std_rating: float,
    ) -> Tuple[Dict[int, float], List[str]]:
        """Extract ``(rated_items, review_texts)`` from an LLM response string.

        Falls back to random ratings on parse failure so ``generate_profiles``
        always returns a valid ``FakeProfile`` regardless of LLM quality.
        """
        target_set     = set(target_items)
        all_items      = list(target_items) + [i for i in filler_items if i not in target_set]

        # ── Extract JSON from the response ────────────────────────────────
        json_text = text.strip()
        # Strip markdown code fences if present
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_text, re.DOTALL)
        if fence:
            json_text = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", json_text, re.DOTALL)
            if brace:
                json_text = brace.group(0)

        parsed_ratings: Dict[int, float] = {}
        parsed_reviews: Dict[int, str]   = {}

        try:
            data        = json.loads(json_text)
            raw_ratings = data.get("ratings", {})
            raw_reviews = data.get("reviews", {})

            for key, val in raw_ratings.items():
                item_str = str(key).replace("item_", "").strip()
                try:
                    item_id = int(item_str)
                    r = float(np.clip(float(val), 1.0, self.max_rating))
                    parsed_ratings[item_id] = round(r * 2) / 2
                except (ValueError, TypeError):
                    pass

            for key, val in raw_reviews.items():
                item_str = str(key).replace("item_", "").strip()
                try:
                    item_id = int(item_str)
                    parsed_reviews[item_id] = str(val).strip()
                except (ValueError, TypeError):
                    pass

        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(
                "Agent4SR: JSON parse failed for response (len=%d); using fallback ratings.",
                len(text),
            )

        # ── Assemble final rated_items and review_texts ───────────────────
        rated: Dict[int, float] = {}
        texts: List[str]        = []

        for item in all_items:
            if item in target_set:
                r = self.max_rating
            elif item in parsed_ratings:
                r = parsed_ratings[item]
            else:
                # Fallback: sample near the dataset mean
                r = float(np.clip(
                    self.rng.normal(mean_rating, max(std_rating, 0.5)),
                    1.0, self.max_rating,
                ))
                r = round(r * 2) / 2
            rated[item] = r

            review = parsed_reviews.get(item, "").strip()
            texts.append(review if review else self._placeholder_text(item, r))

        # Hard-guarantee target items are always at max_rating
        for item in target_items:
            rated[item] = self.max_rating

        return rated, texts

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _cache_path(self, prompt_hash: str) -> Path:
        return self.cache_dir / f"{prompt_hash}.json"

    def _load_cache(self, prompt_hash: str) -> Optional[str]:
        """Return cached LLM response string or None."""
        path = self._cache_path(prompt_hash)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f).get("response")
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        return None

    def _save_cache(self, prompt_hash: str, prompt: str, response: str) -> None:
        """Atomically write ``{prompt, response}`` to the cache directory."""
        path = self._cache_path(prompt_hash)
        tmp  = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {"prompt_hash": prompt_hash, "prompt": prompt, "response": response},
                    f,
                )
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Agent4SR: cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # LLM backend (overridden in LocalAgent4SR)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI Chat Completions API.

        Requires the ``openai`` package and the ``OPENAI_API_KEY`` env var.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for Agent4SRAttack.  "
                "Install it with:  pip install openai"
            ) from exc

        client = openai.OpenAI()           # reads OPENAI_API_KEY from env
        resp   = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # ShillingAttack interface
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Generate *n_fake_users* adversarial profiles via LLM.

        Parameters
        ----------
        n_fake_users : int   — number of fake profiles to generate
        budget_pct   : float — fraction of the item catalog rated per fake user
        target_items : List[int] — global item indices to promote

        Returns
        -------
        List[FakeProfile]
        """
        dataset    = self.dataset
        num_items  = dataset.num_items
        target_set = set(target_items)
        filler_size = max(0, min(round(budget_pct * num_items), MAX_ITEMS_PER_PROFILE) - len(target_items))

        # Dataset statistics for prompt and rating fallback
        mean_r = float(dataset.ratings.mean().item())
        std_r  = float(dataset.ratings.std().item())
        stats  = {
            "name":       dataset.dataset_name,
            "num_users":  dataset.num_users,
            "num_items":  num_items,
            "mean_rating": mean_r,
        }

        non_target_pool = [i for i in range(num_items) if i not in target_set]
        base_ts         = float(dataset.timestamps.max().item())
        profiles: List[FakeProfile] = []

        for uid in range(n_fake_users):
            persona = self.personas[uid % len(self.personas)]

            # Sample filler items for this fake user
            if filler_size > 0 and non_target_pool:
                k            = min(filler_size, len(non_target_pool))
                filler_items = self.rng.choice(non_target_pool, size=k,
                                               replace=False).tolist()
            else:
                filler_items = []

            prompt      = self._build_prompt(stats, target_items, filler_items, persona)
            prompt_hash = self._prompt_hash(prompt)

            # Cache lookup
            cached = self._load_cache(prompt_hash)
            if cached is not None:
                logger.debug("Agent4SR uid=%d: cache hit (%s…)", uid, prompt_hash[:8])
                response = cached
            else:
                logger.debug(
                    "Agent4SR uid=%d: calling %s (persona=%s)",
                    uid, self.model, persona["name"],
                )
                response = self._call_llm(prompt)
                self._save_cache(prompt_hash, prompt, response)

            rated, texts = self._parse_response(
                response, target_items, filler_items, mean_r, std_r
            )

            user_base = base_ts + uid * 1_800.0
            tss       = self._spread_timestamps(len(rated), user_base)

            profiles.append(FakeProfile(
                user_id=f"agent4sr_{uid}",
                rated_items=rated,
                review_texts=texts,
                timestamps=tss,
            ))

        return profiles


# ---------------------------------------------------------------------------
# LocalAgent4SR — Ollama backend
# ---------------------------------------------------------------------------

class LocalAgent4SR(Agent4SRAttack):
    """Agent4SR backed by a local Ollama model (no API key or cloud cost).

    Parameters
    ----------
    dataset  : HeteroDataset
    model    : Ollama model name (default ``"llama3"``)
    host     : Ollama server base URL (default ``"http://localhost:11434"``)
    **kwargs : forwarded to ``Agent4SRAttack.__init__``
    """

    def __init__(
        self,
        dataset: HeteroDataset,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        **kwargs,
    ) -> None:
        super().__init__(dataset, model=model, **kwargs)
        self.host = host.rstrip("/")

    def _call_llm(self, prompt: str) -> str:
        """POST to the Ollama generate API and return the response text."""
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests package is required for LocalAgent4SR.  "
                "Install it with:  pip install requests"
            ) from exc

        url     = f"{self.host}/api/generate"
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
