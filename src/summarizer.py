"""
Main summarizer module.

Orchestrates the full summarization pipeline and dispatches to one of
four modes selected by the caller:

    extractive   — TF-IDF + K-Means (original algorithm)
    abstractive  — sentence fusion with transition phrases
    bullet       — extractive sentences formatted as bullet points
    key_insights — topic-labelled insight cards grouped by top keywords

This is the only module the rest of the app needs to import.
"""

from __future__ import annotations

from src.preprocess import preprocess_text
from src.feature_extraction import build_tfidf_matrix, get_sentence_scores
from src.clustering import cluster_sentences, select_representative_sentences, optimal_k
from src.modes import (
    abstractive_summarize,
    bullet_summarize,
    key_insights_summarize,
    _tfidf_scores,
)

VALID_MODES = {"extractive", "abstractive", "bullet", "key_insights"}


def summarize(
    text: str,
    ratio: float = 0.3,
    mode: str = "extractive",
    auto_k: bool = False,
) -> dict:
    """
    Produce a summary of *text* in the requested *mode*.

    Parameters
    ----------
    text   : str
        The raw input text to summarise.
    ratio  : float, default 0.3
        Fraction of original sentences to keep (0.0 – 1.0).
        Ignored when ``auto_k=True``.
    mode   : str, default "extractive"
        One of "extractive", "abstractive", "bullet", "key_insights".
    auto_k : bool, default False
        If True, use the elbow method to automatically determine the
        optimal number of summary sentences, ignoring ``ratio``.

    Returns
    -------
    dict with keys
        mode                   : str   — the mode used
        summary                : str   — generated summary (extractive / abstractive)
        bullets                : list  — bullet strings (bullet mode only)
        insights               : list  — insight dicts (key_insights mode only)
        selected_indices       : list  — original sentence indices chosen for summary
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
            "selected_indices": list(range(len(sentences))),
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(sentences),
            "compression_ratio": 1.0,
            "sentences": sentences,
            "n_keep": len(sentences),
        }
        return base

    # ── feature extraction (shared by all modes) ───────────────────
    tfidf_matrix, _ = build_tfidf_matrix(sentences)
    scores = _tfidf_scores(sentences)

    # ── determine n_keep (auto vs ratio) ──────────────────────────
    if auto_k:
        n_keep = optimal_k(tfidf_matrix)
    else:
        n_keep = max(1, int(len(sentences) * ratio))
    n_keep = min(n_keep, len(sentences))

    # ── dispatch ──────────────────────────────────────────────────
    if mode == "extractive":
        scores_ex = get_sentence_scores(tfidf_matrix)
        n_clusters = max(1, min(n_keep, len(sentences)))
        labels = cluster_sentences(tfidf_matrix, n_clusters)
        summary_sentences, selected_indices = select_representative_sentences(
            sentences, labels, scores_ex
        )
        summary_text = " ".join(summary_sentences)
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": [],
            "selected_indices": selected_indices,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(summary_sentences),
            "compression_ratio": round(len(summary_sentences) / len(sentences), 2),
            "sentences": sentences,
            "n_keep": n_keep,
        }

    if mode == "abstractive":
        summary_text, selected_indices = abstractive_summarize(sentences, scores, ratio if not auto_k else n_keep / len(sentences))
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": [],
            "selected_indices": selected_indices,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": n_keep,
            "compression_ratio": round(n_keep / len(sentences), 2),
            "sentences": sentences,
            "n_keep": n_keep,
        }

    if mode == "bullet":
        bullets, selected_indices = bullet_summarize(sentences, scores, ratio if not auto_k else n_keep / len(sentences))
        return {
            "mode": mode,
            "summary": " ".join(bullets),
            "bullets": bullets,
            "insights": [],
            "selected_indices": selected_indices,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(bullets),
            "compression_ratio": round(len(bullets) / len(sentences), 2),
            "sentences": sentences,
            "n_keep": n_keep,
        }

    if mode == "key_insights":
        insights, selected_indices = key_insights_summarize(sentences, scores, ratio if not auto_k else n_keep / len(sentences))
        summary_text = " ".join(d["insight"] for d in insights)
        return {
            "mode": mode,
            "summary": summary_text,
            "bullets": [],
            "insights": insights,
            "selected_indices": selected_indices,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(insights),
            "compression_ratio": round(len(insights) / len(sentences), 2),
            "sentences": sentences,
            "n_keep": n_keep,
        }


# ── helpers ────────────────────────────────────────────────────────

def _empty_result(mode: str) -> dict:
    return {
        "mode": mode,
        "summary": "",
        "bullets": [],
        "insights": [],
        "selected_indices": [],
        "original_sentence_count": 0,
        "summary_sentence_count": 0,
        "compression_ratio": 0.0,
    }