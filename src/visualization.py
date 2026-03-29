import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.feature_extraction import build_tfidf_matrix
from src.clustering import cluster_sentences

def generate_cluster_plot(sentences: list[str], n_keep: int):
    """Generate a Plotly PCA scatter plot of sentence clusters."""
    if len(sentences) < 3:
        return None
    
    matrix, _ = build_tfidf_matrix(sentences)
    n_clusters = max(1, min(n_keep, len(sentences)))
    labels = cluster_sentences(matrix, n_clusters)
    
    dense_matrix = matrix.toarray()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(dense_matrix)
    
    # Text wrapping for hover
    hover_texts = []
    for s in sentences:
        wrapped = "<br>".join([s[i:i+80] for i in range(0, len(s), 80)])
        hover_texts.append(wrapped)
        
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Cluster": [f"Cluster {l}" for l in labels],
        "Text": hover_texts
    })
    
    fig = px.scatter(
        df, x="x", y="y", color="Cluster", 
        hover_data={"x": False, "y": False, "Text": True},
        title="Sentence Clusters (PCA of TF-IDF)",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892b0"),
        title_font=dict(size=18, color="#edf0fc"),
        legend_title_font_color="#edf0fc",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    # Hide axes lines and ticks for a cleaner look
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
    
    return fig

def get_top_keywords(sentences: list[str], n_keywords: int = 10) -> list[str]:
    """Extract the top global keywords across all sentences."""
    if not sentences:
        return []
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1, ngram_range=(1, 2))
    mat = vec.fit_transform(sentences)
    feature_names = vec.get_feature_names_out()
    global_scores = np.asarray(mat.sum(axis=0)).flatten()
    top_kw_indices = np.argsort(global_scores)[::-1]
    
    top_keywords = []
    for idx in top_kw_indices:
        kw = feature_names[idx]
        if bool(kw.strip()) and kw not in top_keywords:
            top_keywords.append(kw)
            if len(top_keywords) >= n_keywords:
                break
    return top_keywords
