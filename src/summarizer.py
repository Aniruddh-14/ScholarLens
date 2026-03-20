"""
Main summarizer module.

Orchestrates the full summarization pipeline and dispatches to one of
four modes selected by the caller:

    extractive   — TF-IDF + K-Means (original algorithm)
    abstractive  — sentence fusion with transition phrases
    bullet       — extractive sentences as bullet points
    key_insights — topic-labelled insight cards

This is the only module the rest of the app needs to import.
"""

from __future__ import annotations

from src.preprocess import preprocess_text
from src.feature_extraction import build_tfidf_matrix, get_sentence_scores
from src.clustering import cluster_sentences, select_representative_sentences
from src.modes import (
    abstractive_summarize,
    bullet_summarize,
    key_insights_summarize,
    _tfidf_scores,
)

VALID_MODES = {"extractive", "abstractive", "bullet", "key_insights"}


def summarize(text: str, ratio: float = 0.3, mode: str = "extractive") -> dict:
    """
    Produce a summary of *text* in the requested *mode*.

    Parameters
    ----------
    text  : str
        The raw input text to summarise.
    ratio : float, default 0.3
        Fraction of original sentences to keep (0.0 – 1.0).
    mode  : str, default "extractive"
        One of "extractive", "abstractive", "bullet", "key_insights".

    Returns
    -------
    dict with keys
        mode                   : str   — the mode used
        summary                : str   — generated summary (extractive / abstractive)
        bullets                : list  — bullet strings (bullet mode only)
        insights               : list  — insight dicts (key_insights mode only)
        original_sentence_count: int
        summary_sentence_count : int
        compression_ratio      : float
    """
    if mode not in VALID_MODES:
        mode = "extractive"

    if not text or not text.strip():
        return _empty_result(mode)

    cleaned, sentences = preprocess_text(text)

    if len(sentences) <= 2:
        base = {
            "mode": mode,
            "summary": cleaned,
            "bullets": [s for s in sentences],
            "insights": [{"topic": "Key Point", "insight": s} for s in sentences],
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(sentences),
            "compression_ratio": 1.0,
        }
        return base

    # ── extractive core (shared by all modes for scoring) ─────────
    scores = _tfidf_scores(sentences)
    n_keep = max(1, int(len(sentences) * ratio))

    # ── dispatch ──────────────────────────────────────────────────
    if mode == "extractive":
        tfidf_matrix, _ = build_tfidf_matrix(sentences)
        scores_ex = get_sentence_scores(tfidf_matrix)
        n_clusters = max(1, min(n_keep, len(sentences)))
        labels = cluster_sentences(tfidf_matrix, n_clusters)
        summary_sentences = select_representative_sentences(sentences, labels, scores_ex)
        summary_text = " ".join(summary_sentences)
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": [],
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(summary_sentences),
            "compression_ratio": round(len(summary_sentences) / len(sentences), 2),
        }

    if mode == "abstractive":
        summary_text = abstractive_summarize(sentences, scores, ratio)
        abs_sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": [],
            "original_sentence_count": len(sentences),
            "summary_sentence_count": n_keep,
            "compression_ratio": round(n_keep / len(sentences), 2),
        }

    if mode == "bullet":
        bullets = bullet_summarize(sentences, scores, ratio)
        return {
            "mode": mode,
            "summary": " ".join(bullets),
            "bullets": bullets,
            "insights": [],
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(bullets),
            "compression_ratio": round(len(bullets) / len(sentences), 2),
        }

    if mode == "key_insights":
        insights = key_insights_summarize(sentences, scores, ratio)
        summary_text = " ".join(d["insight"] for d in insights)
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": insights,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(insights),
            "compression_ratio": round(len(insights) / len(sentences), 2),
        }


# ── helpers ────────────────────────────────────────────────────────

def _empty_result(mode: str) -> dict:
    return {
        "mode": mode,
        "summary": "",
        "bullets": [],
        "insights": [],
        "original_sentence_count": 0,
        "summary_sentence_count": 0,
        "compression_ratio": 0.0,
    }