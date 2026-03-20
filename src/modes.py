"""
Alternative summarization modes for ScholarLens.

Provides three extra modes that all sit on top of the same extractive
sentence-selection core so no extra heavy dependencies are required:

  abstractive  — TF-IDF-guided sentence fusion with transition phrases
  bullet       — extractive sentences formatted as bullet points
  key_insights — topic-labelled insight cards grouped by top keywords
"""

from __future__ import annotations

import re
import math
import random
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import preprocess_text, tokenize_and_clean


# ── shared helpers ─────────────────────────────────────────────────

def _tfidf_scores(sentences: list[str]) -> np.ndarray:
    """Return a per-sentence mean-TF-IDF importance score array."""
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
    mat = vec.fit_transform(sentences)
    return np.asarray(mat.mean(axis=1)).flatten()


def _top_sentences(sentences: list[str], scores: np.ndarray, n: int) -> list[tuple[int, str]]:
    """Return up to *n* (original_index, sentence) pairs, scored highest first."""
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    # keep the top n, then re-sort by document order
    top = sorted(ranked[:n], key=lambda x: x[0])
    return [(i, sentences[i]) for i, _ in top]


# ── abstractive mode ───────────────────────────────────────────────

# Short transition phrases injected between fused sentences
_TRANSITIONS = [
    "Furthermore,", "Additionally,", "Notably,",
    "In particular,", "Moreover,", "As a result,",
    "Consequently,", "This means that", "In other words,",
]

_STOP = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "and", "or",
    "but", "in", "on", "at", "to", "of", "for", "with", "by",
    "as", "from", "not", "no", "so", "if", "then", "also",
])


def _trim_sentence(sentence: str, keep_ratio: float = 0.75) -> str:
    """
    Lightly shorten a sentence by dropping trailing low-importance clauses.

    Strategy:
      Split on commas / semicolons / relative-clause markers.
      Drop trailing fragments until we hit keep_ratio of original length.
    """
    # Split on minor clause boundaries
    parts = re.split(r"(?<=\w)[,;]\s+(?=[a-z])", sentence)
    if len(parts) == 1:
        return sentence

    target_len = max(1, math.ceil(len(parts) * keep_ratio))
    trimmed = " ".join(parts[:target_len])

    # Ensure it ends with proper punctuation
    if trimmed and trimmed[-1] not in ".!?":
        trimmed = trimmed.rstrip(",;:") + "."
    return trimmed


def abstractive_summarize(
    sentences: list[str],
    scores: np.ndarray,
    ratio: float,
) -> str:
    """
    Lightweight sentence-fusion abstractive summarizer.

    Steps
    -----
    1. Select top-scored sentences (same as extractive).
    2. Lightly trim each sentence to remove trailing low-value clauses.
    3. Insert varied transition phrases between sentences so the result
       reads as a flowing paragraph rather than extracted fragments.
    """
    n = max(1, int(len(sentences) * ratio))
    selected = _top_sentences(sentences, scores, n)

    rng = random.Random(42)  # deterministic transitions
    parts: list[str] = []
    for idx, (_, sent) in enumerate(selected):
        trimmed = _trim_sentence(sent)
        if idx == 0:
            parts.append(trimmed)
        else:
            bridge = rng.choice(_TRANSITIONS)
            # lower-case first word of trimmed if bridge ends with a space
            body = trimmed[0].lower() + trimmed[1:] if trimmed else trimmed
            parts.append(f"{bridge} {body}")

    return " ".join(parts)


# ── bullet mode ────────────────────────────────────────────────────

def bullet_summarize(
    sentences: list[str],
    scores: np.ndarray,
    ratio: float,
) -> list[str]:
    """
    Return a list of bullet-point strings (without the bullet character).
    Each item is one key sentence, lightly cleaned.
    """
    n = max(1, int(len(sentences) * ratio))
    selected = _top_sentences(sentences, scores, n)
    bullets = []
    for _, sent in selected:
        clean = sent.strip()
        if clean and clean[-1] not in ".!?":
            clean += "."
        bullets.append(clean)
    return bullets


# ── key insights mode ──────────────────────────────────────────────

def key_insights_summarize(
    sentences: list[str],
    scores: np.ndarray,
    ratio: float,
    n_topics: int = 5,
) -> list[dict]:
    """
    Group the most important sentences under labelled topic headings.

    Returns
    -------
    list of dicts: [{"topic": str, "insight": str}, ...]
    """
    # Step 1 — pick the top-scored sentences to work with
    n = max(2, int(len(sentences) * min(ratio * 1.5, 1.0)))
    selected = _top_sentences(sentences, scores, n)

    # Step 2 — extract top global keywords via TF-IDF on all sentences
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1,
                          ngram_range=(1, 2))
    mat = vec.fit_transform(sentences)
    feature_names = vec.get_feature_names_out()
    global_scores = np.asarray(mat.sum(axis=0)).flatten()
    top_kw_indices = np.argsort(global_scores)[::-1][:n_topics * 3]
    top_keywords = [feature_names[i] for i in top_kw_indices
                    if " " not in feature_names[i]][:n_topics]

    # Step 3 — for each keyword, find the selected sentence that contains it
    used_sentences: set[int] = set()
    insights: list[dict] = []

    for kw in top_keywords:
        best_idx = None
        best_score = -1.0
        for orig_idx, sent in selected:
            if orig_idx in used_sentences:
                continue
            if kw.lower() in sent.lower():
                if scores[orig_idx] > best_score:
                    best_score = scores[orig_idx]
                    best_idx = orig_idx
        if best_idx is None:
            continue
        used_sentences.add(best_idx)
        insight_text = sentences[best_idx].strip()
        if insight_text and insight_text[-1] not in ".!?":
            insight_text += "."
        insights.append({
            "topic": kw.replace("-", " ").title(),
            "insight": insight_text,
        })
        if len(insights) >= n_topics:
            break

    # Step 4 — if we still have unclaimed selected sentences, add as "Key Point"
    for orig_idx, sent in selected:
        if orig_idx not in used_sentences and len(insights) < n_topics:
            used_sentences.add(orig_idx)
            text = sent.strip()
            if text and text[-1] not in ".!?":
                text += "."
            insights.append({"topic": "Key Point", "insight": text})

    return insights
