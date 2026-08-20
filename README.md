Information Retrieval Assignment 2 – Group 57

This project is a single Streamlit-based Information Retrieval system using the Amazon Camera/Electronics customer-review corpus.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

NLTK downloads `stopwords` and `wordnet` on first launch. Internet access is required for the live crawler and the first NLTK download.

## Run

```bash
streamlit run app.py
```

Use the Streamlit interface only:

1. Dashboard & Index Management: inspect corpus/index/graph status and rebuild the index.
2. Ingestion & Text Mining: crawl multiple seed URLs with configurable depth, inspect duplicate-safe crawl output, preprocess text, profile documents, classify categories, and compare preprocessing strategies.
3. Web Search & Graph Ranking: search the inverted index with BM25 and PageRank-aware ranking. The result table and chart show how PageRank changes ranking.
4. Recommenders & IR Evaluation: generate Top-K content-based or collaborative recommendations and evaluate retrieval using Precision, Recall, F1-score, Precision@K, Recall@K, MAP, MRR, and NDCG, including comparative BM25/PageRank analysis.

## Data

The included `data/sample_amazon_reviews1.csv` is the runtime dataset. `data/sample_amazon_reviews.csv` is the larger supporting dataset. The dataset is derived from the Amazon US Customer Reviews public dataset and contains product IDs, customer IDs, titles, review text, ratings, and categories.

## Submission evidence

Run every Streamlit module in the BITS Virtual Lab and capture screenshots or a short recording showing the dashboard, crawler, text-mining outputs, search/PageRank results, recommendations, and evaluation comparison. Include those artifacts with `REPORT.md` in the final submission.
