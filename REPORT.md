# Information Retrieval Assignment 2 – Report

## 1. Use case and data

The system retrieves and recommends Amazon products from customer-review text. The collection contains product metadata, customer identifiers, product categories, star ratings, and review bodies. The Streamlit application provides the complete workflow from ingestion and preprocessing through indexing, search, graph ranking, recommendation, and evaluation.

## 2. Implementation

- Crawling: multiple seed URLs, configurable depth/page limit, canonical URL tracking, exact duplicate-document hashing, and separate metadata/content records.
- Text mining: normalization, tokenization, stop-word removal, optional lemmatization, n-grams, TF-IDF features, word-cloud visualization, document profiling, category distribution, and category classification.
- Search and ranking: an inverted index with BM25 retrieval over aggregated product documents, followed by PageRank-aware re-ranking on a co-review product graph.
- Recommendation: content-based TF-IDF/cosine recommendations and collaborative user-item matrix-factorization recommendations with Top-K scores.
- Evaluation: Precision, Recall, F1-score, Precision@K, Recall@K, MAP, MRR, and NDCG, with a BM25 versus BM25+PageRank comparison table and chart.

## 3. Experimental results

Run the application in the Virtual Lab and paste the output tables and screenshots here. Use the multi-query evaluation box with at least three queries and repeat at two K values. Record the relevance definition, retrieved results, metrics, and timing. The displayed MAP is the mean of the per-query average precision values.

| Query | K | Ranking | Precision@K | Recall@K | F1@K | MAP | MRR | NDCG@K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| camera lens | 5 | BM25 |  |  |  |  |  |  |
| camera lens | 5 | BM25 + PageRank |  |  |  |  |  |  |

## 4. Mandatory inference and discussion

### 4.1 Relevant documents are retrieved but ranked poorly

This can occur when term matching gives similar BM25 scores to many documents, when document length normalization is unsuitable, when the query is ambiguous, or when the index contains duplicate/low-quality documents. Ranking can be improved with a graph-authority signal such as PageRank, better query preprocessing, field weighting for titles, duplicate removal, and tuning the BM25/PageRank combination on validation queries. In this project, PageRank is used as the structural ranking signal and its effect is displayed in the Streamlit search results.

### 4.2 Duplicate and near-duplicate documents

Duplicates inflate term frequencies, consume index space, create repeated search results, bias recommendation similarity, and can make evaluation appear better because the same information is counted multiple times. Exact duplicates are mitigated with canonical URL tracking and content hashes. Near-duplicates should be identified using normalized text fingerprints or MinHash/LSH and either removed or clustered before indexing and evaluation.

### 4.3 Content-based versus collaborative recommendation

Content-based recommendation is preferable when item metadata/text is available or when users are new, because it can recommend items without a large interaction history and is explainable through similarity. Collaborative recommendation is preferable when many users have overlapping interactions, because it can discover relationships that are not obvious from text. It suffers from cold-start and sparsity. The application exposes both approaches so their Top-K scores can be inspected.

### 4.4 End-to-end integration

Crawling and dataset ingestion supply the collection. Preprocessing converts noisy text into consistent tokens and features. Indexing makes retrieval efficient. BM25 finds textually relevant candidates, while PageRank adds structural authority from the product graph. Recommendations reuse content and user-interaction signals for personalization. Evaluation measures retrieval quality and exposes trade-offs. Together these stages create a repeatable IR lifecycle rather than isolated notebook outputs.

### 4.5 Learnings from the results

The main learning should be based on the recorded Virtual Lab results. Compare whether PageRank changes the top results and whether NDCG/MRR improve or decline. Explain how preprocessing changes vocabulary size and document length, how category classification performs, and whether content-based or collaborative recommendations are more useful for this dataset. Discuss limitations including pseudo relevance judgments, sparse user-item interactions, crawler coverage, and duplicate handling.

## 5. Virtual Lab and demo evidence

Insert screenshots or a short recording showing the running Streamlit application. The evidence must cover:

1. Dashboard/index status.
2. Multiple-seed crawler and configurable depth.
3. Text preprocessing, profiling, classification, and visualizations.
4. Search results with PageRank-aware ranking.
5. Top-K recommendations with scores.
6. Evaluation metrics and comparative analysis.
