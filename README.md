# dataset chosen
**amazon-us-customer-reviews** - https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

# Information Retrieval (IR) Assignment 2 - Work Breakdown & Team Allocation

This document outlines the task distribution, system architecture, and integration roadmap for 3 team members collaborating on **IR Assignment 2** using the **Amazon US Customer Reviews Dataset**.

---

## Task Distribution Matrix

### Person 1: Data Acquisition, Crawling, Preprocessing & Text Mining

* **Primary Focus:** Data ingestion, web crawling, duplicate handling, text cleaning, feature engineering, and corpus analytics.
* **Core Deliverables:**
  * **Crawler & Ingestion Module (Section B):**
    * Develop a live product page crawler/scraper (using `BeautifulSoup` or `Scrapy`) with configurable crawling depth and seed sources.
    * Implement duplicate detection logic for URLs and documents (e.g., MinHash, LSH, or MD5 hashing).
    * Separate extracted metadata (product IDs, ratings, categories) from raw text contents[cite: 1].
  * **Text Preprocessing Framework (Section C):**
    * Implement text normalization, tokenization, stop-word removal, and stemming/lemmatization[cite: 1].
    * Support feature extraction methods (TF-IDF, Bag-of-Words, or Word Embeddings)[cite: 1].
  * **Text Analytics Dashboard (Section C):**
    * Build Streamlit visualizations for word clouds, n-gram distributions, category classification, and comparative preprocessing analyses[cite: 1].

---

### Person 2: Indexing, Web Search Engine & Graph Ranking

* **Primary Focus:** Inverted index construction, search interface, query processing, and graph-based ranking algorithms[cite: 1].
* **Core Deliverables:**
  * **Index Management & Query Engine (Section D):**
    * Build efficient inverted indexes over product titles and review contents[cite: 1].
    * Implement keyword search and vector space retrieval (Cosine Similarity, BM25) with query optimization[cite: 1].
  * **Graph Ranking Engine (Section D):**
    * Construct a hyperlinked product graph (e.g., products co-reviewed by the same users or linked by category hierarchies)[cite: 1].
    * Implement **PageRank** or **HITS** to compute structural authority scores[cite: 1].
  * **Search UI & Ranking Visualizations (Section A & D):**
    * Design the Streamlit search interface displaying ranked search hits with metadata (product ID, star ratings, category)[cite: 1].
    * Render visual graphs demonstrating how PageRank/HITS re-ranks retrieval results[cite: 1].

---

### Person 3: Recommender System, IR Evaluation & Final Inferences

* **Primary Focus:** Personalization engines, system performance evaluation, and mandatory report synthesis[cite: 1].
* **Core Deliverables:**
  * **Recommender Systems Panel (Section E):**
    * Build a **Content-Based Filtering** model (TF-IDF / Cosine Similarity on product text)[cite: 1].
    * Build a **Collaborative Filtering** or **Hybrid** model using user-item rating matrices[cite: 1].
    * Display Top-$K$ recommended items along with explicit similarity scores[cite: 1].
  * **IR Evaluation Dashboard (Section F):**
    * Compute standard IR metrics: **Precision@K, Recall@K, F1-Score, MAP, MRR, and NDCG**[cite: 1].
    * Provide comparative tables and interactive performance charts in Streamlit[cite: 1].
  * **Report & Discussion Questions (Section G):**
    * Write comprehensive answers for all 5 mandatory discussion questions[cite: 1].
    * Assemble the final project report, demo video/screenshots, README setup guide, and Virtual Lab submission package[cite: 1].

---

## Technical Architecture & Responsibilities Summary

| Team Member | Streamlit UI Pages Owned[cite: 1] | Key Python Libraries & Algorithms |
| :--- | :--- | :--- |
| **Person 1** | Crawling Interface, Text Mining Dashboard[cite: 1] | `BeautifulSoup`, `NLTK` / `spaCy`, `scikit-learn`, `Plotly` |
| **Person 2** | Search Engine UI, Indexing & Graph Ranking Visualizer[cite: 1] | `NetworkX` (PageRank/HITS), `Whoosh` or custom Inverted Index, `rank_bm25` |
| **Person 3** | Recommender Panel, IR Evaluation Dashboard[cite: 1] | `scikit-learn`, `Surprise` (SVD/Cosine), IR evaluation metrics |

---

## Streamlit Application Directory Structure

To enable parallel development, use Streamlit's multi-page application structure:

```text
ir_assignment2_group_57/
├── app.py                      # Main entry point (Streamlit Landing Page)
├── requirements.txt            # Dependency list
├── README.md                   # Setup and execution instructions
├── data/
│   ├── sample_amazon_reviews.csv
│   └── inverted_index.pkl
├── pages/
│   ├── 1_Data_Mining.py        # Owned by Person 1
│   ├── 2_Search_Engine.py      # Owned by Person 2
│   └── 3_Recommendations.py    # Owned by Person 3
└── utils/
    ├── crawler.py              # Person 1
    ├── indexer.py              # Person 2
    ├── ranker.py               # Person 2
    ├── recommenders.py         # Person 3
    └── evaluation.py           # Person 3
