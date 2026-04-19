"""
Sentence clustering module.

Groups sentences by topical similarity using K-Means and picks the most
informative representative from each cluster.

Math refresher
--------------
K-Means minimises the *within-cluster sum of squares* (WCSS / inertia):

    J = Σ_{k=1}^{K}  Σ_{x ∈ C_k}  ‖x − μ_k‖²

where μ_k is the centroid of cluster C_k.

We pick *k* from user-specified `ratio` (fraction of sentences to keep),
but also expose `optimal_k()` which uses the *elbow method* — finding the
k where diminishing returns on J start to flatten out.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# ── Cluster assignment ─────────────────────────────────────────────

def cluster_sentences(
    tfidf_matrix,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run K-Means on the TF-IDF sentence vectors.

    Parameters
    ----------
    tfidf_matrix  : sparse/dense matrix, shape (n_sentences, n_features)
    n_clusters    : number of clusters (= desired summary sentences)
    random_state  : seed for reproducibility

    Returns
    -------
    labels : ndarray of shape (n_sentences,)
        Cluster id for each sentence.
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    return km.fit_predict(tfidf_matrix)


# ── Optimal k via the elbow heuristic ──────────────────────────────

def optimal_k(
    tfidf_matrix,
    max_k: int | None = None,
    random_state: int = 42,
) -> int:
    """
    Estimate a reasonable number of clusters using the *elbow method*.

    We compute inertia for k = 2 … max_k, then locate the "elbow" —
    the k with the largest second-derivative (biggest rate-of-change drop).

    Parameters
    ----------
    tfidf_matrix : sparse/dense matrix
    max_k        : upper bound for k (defaults to n_sentences // 2)
    random_state : seed

    Returns
    -------
    best_k : int
        Suggested number of clusters (>= 2).
    """
    n = tfidf_matrix.shape[0]
    if n <= 3:
        return max(1, n - 1)

    if max_k is None:
        max_k = max(3, n // 2)
    max_k = min(max_k, n - 1)  # can't have more clusters than sentences

    ks = list(range(2, max_k + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=200)
        km.fit(tfidf_matrix)
        inertias.append(km.inertia_)

    # second-derivative approach to find the elbow
    if len(inertias) < 3:
        return ks[0]

    diffs_1 = np.diff(inertias)          # first derivative
    diffs_2 = np.diff(diffs_1)           # second derivative
    # elbow = point with max second derivative (biggest curvature change)
    elbow_idx = int(np.argmax(diffs_2)) + 2   # +2 because of two np.diff calls
    elbow_idx = min(elbow_idx, len(ks) - 1)
    return ks[elbow_idx]


# ── Representative selection ───────────────────────────────────────

def select_representative_sentences(
    sentences: list[str],
    labels: np.ndarray,
    sentence_scores: np.ndarray,
) -> tuple[list[str], list[int]]:
    """
    From each cluster pick the sentence with the highest TF-IDF score,
    then return them in their original document order.

    This preserves narrative flow — the summary reads like a natural
    subset of the original text rather than a shuffled bag of facts.

    Parameters
    ----------
    sentences       : original sentence list
    labels          : cluster id per sentence (from `cluster_sentences`)
    sentence_scores : per-sentence TF-IDF importance score

    Returns
    -------
    tuple[list[str], list[int]]
        - Selected sentences, ordered by their position in the document.
        - Corresponding original indices of the selected sentences.
    """
    selected_indices: list[int] = []

    for cluster_id in range(labels.max() + 1):
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]

        if len(member_indices) == 0:
            continue  # empty cluster (rare, but possible)

        # pick the highest-scoring member
        best = member_indices[np.argmax(sentence_scores[member_indices])]
        selected_indices.append(int(best))

    # sort by position so the summary follows original order
    selected_indices.sort()
    return [sentences[i] for i in selected_indices], selected_indices