# 📝 ScholarLens

**ScholarLens** is an extractive text summarization tool built with Python. It uses **TF-IDF** feature extraction and **K-Means clustering** to intelligently select the most informative sentences from any document — no neural networks required.

> **Live demo:** Hosted on [Streamlit Community Cloud](https://aiml-project.streamlit.app)

---

## How It Works

```
Raw Text  →  Preprocess  →  TF-IDF Features  →  K-Means Clustering  →  Summary
```

### 1. Preprocessing
- Collapse whitespace and strip noise characters
- Split text into sentences using NLTK's Punkt tokenizer
- Tokenize words and remove stopwords

### 2. Feature Extraction (TF-IDF)
Each sentence is converted into a numeric vector using **Term Frequency – Inverse Document Frequency**:

```
tf(t, d)    = count of term t in sentence d  /  total terms in d
idf(t, D)   = log( N / (1 + df(t)) ) + 1
tfidf(t, d) = tf(t, d) × idf(t, D)
```

- **Sublinear TF** (`1 + log(tf)`) dampens very frequent words
- **L2 row-normalisation** gives each sentence a unit-length vector
- Each sentence's **importance score** = mean of its TF-IDF values

### 3. K-Means Clustering
Sentences are grouped into `k` clusters (`k = ratio × n_sentences`), minimising within-cluster sum of squares to group topically similar sentences.

### 4. Representative Selection
The highest-scoring sentence from each cluster is selected, then sorted by original position to preserve narrative flow.

---

## Project Structure

```
ScholarLens/
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
└── src/
    ├── __init__.py             # Package docstring
    ├── preprocess.py           # Text cleaning & sentence splitting
    ├── feature_extraction.py   # TF-IDF matrix + sentence scoring
    ├── clustering.py           # K-Means clustering + elbow method
    ├── summarizer.py           # Orchestrates the full pipeline
    └── utils.py                # Helper functions & sample texts
```

---

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/Aniruddh-14/ScholarLens.git
cd ScholarLens

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select this repo → branch `main` → file `app.py`.
4. Click **Deploy**.

Streamlit Cloud auto-installs packages from `requirements.txt` and picks up `.streamlit/config.toml`.

---

## Tech Stack

| Library | Purpose |
|---|---|
| **Streamlit** | Interactive web UI with dark/light theme |
| **NLTK** | Sentence tokenization and stopword lists |
| **scikit-learn** | TF-IDF vectorization and K-Means clustering |
| **NumPy** | Numerical operations |

---

## License

This project is for educational and research purposes.
