import hashlib
import math
import re
import time
from collections import defaultdict
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import streamlit as st
from wordcloud import WordCloud

# Ensure NLTK data dependencies
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ------------------------------------------------------------------------------
# 1. CORE ALGORITHMIC UTILITIES
# ------------------------------------------------------------------------------


class SimpleCrawler:

  def __init__(self, max_depth=2, max_pages=15):
    self.max_depth = max_depth
    self.max_pages = max_pages
    self.visited_urls = set()
    self.doc_hashes = set()
    self.crawled_data = []

  def _get_hash(self, content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()

  @staticmethod
  def _canonical_url(url):
    parsed = urlparse(url.strip())
    path = parsed.path or "/"
    return parsed._replace(fragment="", query="", path=path.rstrip("/") or "/").geturl()

  def crawl(self, seed_urls):
    queue = [(self._canonical_url(url), 0) for url in seed_urls if url.strip()]
    while queue and len(self.crawled_data) < self.max_pages:
      url, depth = queue.pop(0)
      url = self._canonical_url(url)
      if url in self.visited_urls or depth > self.max_depth:
        continue
      self.visited_urls.add(url)
      try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (IR_Assignment_Bot)"},
        )
        if response.status_code != 200:
          continue
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator=" ", strip=True)
        doc_hash = self._get_hash(text_content)
        if doc_hash in self.doc_hashes:
          continue
        self.doc_hashes.add(doc_hash)

        title = soup.title.string if soup.title else url
        metadata = {
            "url": url,
            "title": title,
            "depth": depth,
            "domain": urlparse(url).netloc,
        }
        self.crawled_data.append(
            {"metadata": metadata, "content": text_content}
        )

        if depth < self.max_depth:
          for link in soup.find_all("a", href=True):
            next_url = self._canonical_url(urljoin(url, link["href"]))
            if (
                next_url.startswith("http")
                and next_url not in self.visited_urls
            ):
              queue.append((next_url, depth + 1))
      except Exception:
        continue
    return self.crawled_data


class TextMiner:

  def __init__(self):
    try:
      self.stop_words = set(stopwords.words("english"))
    except LookupError:
      self.stop_words = set()
    self.lemmatizer = WordNetLemmatizer()

  def preprocess(self, text, use_lemmatization=True):
    if not isinstance(text, str):
      return ""
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
    if use_lemmatization:
      try:
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
      except LookupError:
        pass
    return " ".join(tokens)

  def get_top_ngrams(self, corpus, n=2, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=["ngram", "frequency"])

  def extract_tfidf_features(self, corpus, max_features=20):
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names_out()
    mean_tfidf = np.asarray(matrix.mean(axis=0)).ravel()
    return pd.DataFrame(
        {"feature": feature_names, "mean_tfidf": mean_tfidf}
    ).sort_values(by="mean_tfidf", ascending=False)

  def document_profile(self, df, text_col="review_body"):
    profile = df.copy()
    profile["word_count"] = profile[text_col].fillna("").astype(str).str.split().str.len()
    profile["char_count"] = profile[text_col].fillna("").astype(str).str.len()
    return profile

  def classify_documents(self, df):
    """Classify documents into the dataset's product categories."""
    work = df[["review_body", "product_category"]].dropna()
    if work["product_category"].nunique() < 2 or len(work) < 20:
      return {"accuracy": 0.0, "classes": int(work["product_category"].nunique())}
    x_train, x_test, y_train, y_test = train_test_split(
        work["review_body"], work["product_category"], test_size=0.2,
        random_state=42, stratify=work["product_category"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("classifier", LogisticRegression(max_iter=300)),
    ])
    model.fit(x_train, y_train)
    return {"accuracy": float(accuracy_score(y_test, model.predict(x_test))),
            "classes": int(work["product_category"].nunique())}


class InvertedIndex:

  def __init__(self):
    self.index = defaultdict(list)
    self.doc_lengths = {}
    self.avg_doc_len = 0
    self.corpus = {}

  def _tokenize(self, text):
    if not isinstance(text, str):
      return []
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

  def build_index(self, df, id_col="product_id", text_col="review_body"):
    grouped = df.groupby(id_col, as_index=False).agg(
        {text_col: lambda s: " ".join(s.astype(str)),
         "product_title": "first", "star_rating": "mean",
         "product_category": "first"}
    )
    total_len = 0
    for _, row in grouped.iterrows():
      doc_id = row[id_col]
      tokens = self._tokenize(str(row[text_col]) + " " + str(row["product_title"]))
      if doc_id not in self.corpus:
        self.corpus[doc_id] = row.to_dict()
        self.doc_lengths[doc_id] = len(tokens)
        total_len += len(tokens)

      tf_counts = defaultdict(int)
      for token in tokens:
        tf_counts[token] += 1
      for token, count in tf_counts.items():
        self.index[token].append((doc_id, count))
    self.avg_doc_len = total_len / max(len(self.corpus), 1)

  def search_bm25(self, query, k1=1.5, b=0.75, top_k=20):
    query_tokens = self._tokenize(query)
    scores = defaultdict(float)
    N = len(self.doc_lengths)
    for token in query_tokens:
      if token not in self.index:
        continue
      postings = self.index[token]
      df = len(postings)
      idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
      for doc_id, tf in postings:
        doc_len = self.doc_lengths[doc_id]
        denom = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
        scores[doc_id] += idf * ((tf * (k1 + 1)) / denom)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    return [(self.corpus[doc_id], score) for doc_id, score in sorted_docs]


class GraphRanker:

  def __init__(self):
    self.graph = nx.DiGraph()

  def build_co_review_graph(self, df):
    user_to_products = defaultdict(set)
    for _, row in df.iterrows():
      user_to_products[row["customer_id"]].add(row["product_id"])

    for products in user_to_products.values():
      prod_list = list(products)
      for i in range(len(prod_list)):
        for j in range(i + 1, len(prod_list)):
          self.graph.add_edge(prod_list[i], prod_list[j])
          self.graph.add_edge(prod_list[j], prod_list[i])

  def compute_pagerank(self, alpha=0.85):
    return nx.pagerank(self.graph, alpha=alpha) if len(self.graph) > 0 else {}

class RecommenderEngine:

  def __init__(self, df):
    self.df = df.dropna(subset=["product_id", "customer_id", "review_body"])
    self._build_content_model()
    self._build_collaborative_model()

  def _build_content_model(self):
    self.df["content_mix"] = (
        self.df["product_title"].fillna("")
        + " "
        + self.df["review_body"].fillna("")
    )
    self.product_df = self.df.drop_duplicates(subset=["product_id"]).reset_index(
        drop=True
    )
    self.product_ids = self.product_df["product_id"].tolist()
    self.pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(self.product_df["content_mix"])
    self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

  def _build_collaborative_model(self):
    user_item_matrix = self.df.pivot_table(
        index="customer_id", columns="product_id", values="star_rating"
    ).fillna(0)
    self.user_ids = user_item_matrix.index.tolist()
    self.collab_product_ids = user_item_matrix.columns.tolist()

    R = user_item_matrix.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    k = min(10, min(R.shape) - 1)
    if k > 0:
      U, sigma, Vt = svds(R_demeaned, k=k)
      sigma = np.diag(sigma)
      predicted_ratings = (
          np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
      )
      self.cf_df = pd.DataFrame(
          predicted_ratings,
          index=self.user_ids,
          columns=self.collab_product_ids,
      )
    else:
      self.cf_df = pd.DataFrame(
          R, index=self.user_ids, columns=self.collab_product_ids
      )

  def get_content_recommendations(self, product_id, top_k=5):
    if product_id not in self.pid_to_idx:
      return []
    idx = self.pid_to_idx[product_id]
    sim_scores = list(enumerate(self.content_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
        1 : top_k + 1
    ]

    recs = []
    for i, score in sim_scores:
      row = self.product_df.iloc[i]
      recs.append({
          "Product ID": row["product_id"],
          "Product Title": row["product_title"],
          "Similarity Score": round(score, 4),
      })
    return recs

  def get_collaborative_recommendations(self, customer_id, top_k=5):
    if customer_id not in self.cf_df.index:
      return []
    rated = set(self.df.loc[self.df["customer_id"] == customer_id, "product_id"])
    user_preds = self.cf_df.loc[customer_id].drop(index=rated, errors="ignore") \
        .sort_values(ascending=False).head(top_k)

    recs = []
    for pid, predicted_score in user_preds.items():
      title_match = self.product_df[self.product_df["product_id"] == pid][
        "product_title"
      ]
      title = (
          title_match.values[0]
          if len(title_match) > 0
          else "Unknown Product Title"
      )
      recs.append({
          "Product ID": pid,
          "Product Title": title,
          "Predicted Rating": round(predicted_score, 4),
      })
    return recs


class IREvaluator:

  @staticmethod
  def evaluate_retrieval(retrieved_ids, pseudo_relevant_set, k=5):
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in pseudo_relevant_set]

    p_full = (
        len([doc for doc in retrieved_ids if doc in pseudo_relevant_set]) /
        len(retrieved_ids)
        if retrieved_ids else 0.0
    )
    r_full = (
        len(set(retrieved_ids).intersection(pseudo_relevant_set)) /
        len(pseudo_relevant_set)
        if pseudo_relevant_set else 0.0
    )
    p_k = len(relevant_retrieved) / k if k > 0 else 0.0
    r_k = (
        len(relevant_retrieved) / len(pseudo_relevant_set)
        if pseudo_relevant_set
        else 0.0
    )
    f1 = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0

    mrr = 0.0
    for rank, doc in enumerate(retrieved_ids, start=1):
      if doc in pseudo_relevant_set:
        mrr = 1.0 / rank
        break

    hits = 0
    sum_prec = 0.0
    for i, doc in enumerate(retrieved_ids, start=1):
      if doc in pseudo_relevant_set:
        hits += 1
        sum_prec += hits / i
    map_score = sum_prec / len(pseudo_relevant_set) if pseudo_relevant_set else 0.0

    dcg = sum([
        1.0 / np.log2(i + 1)
        for i, doc in enumerate(retrieved_k, start=1)
        if doc in pseudo_relevant_set
    ])
    idcg = sum([
        1.0 / np.log2(i + 1)
        for i in range(1, min(len(pseudo_relevant_set), k) + 1)
    ])
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "Precision": round(p_full, 4),
        "Recall": round(r_full, 4),
        "F1-score": round((2 * p_full * r_full) / (p_full + r_full), 4)
        if p_full + r_full else 0.0,
        f"Precision@{k}": round(p_k, 4),
        f"Recall@{k}": round(r_k, 4),
        f"F1-Score@{k}": round(f1, 4),
        "MRR": round(mrr, 4),
        "MAP": round(map_score, 4),
        f"NDCG@{k}": round(ndcg, 4),
    }

  @staticmethod
  def evaluate_queries(query_results, k=5):
    """Return per-query metrics and their mean across queries.

    query_results is an iterable of (query, retrieved_ids, relevant_ids).
    The mean of per-query average precision values is true MAP.
    """
    rows = []
    for query, retrieved_ids, relevant_ids in query_results:
      rows.append({"Query": query, **IREvaluator.evaluate_retrieval(
          retrieved_ids, set(relevant_ids), k=k
      )})
    detail = pd.DataFrame(rows)
    if detail.empty:
      return detail, pd.DataFrame()
    numeric = detail.select_dtypes(include="number").mean().to_frame().T
    numeric.insert(0, "Query", "Mean across queries")
    return detail, numeric


# ------------------------------------------------------------------------------
# 2. UNIFIED STREAMLIT APPLICATION
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Information Retrieval System", page_icon="🔍", layout="wide"
)

st.title("Unified Information Retrieval & Recommender System")
st.caption("BITS Pilani WILP - Assignment 2")


def build_system(df):
  idx = InvertedIndex()
  idx.build_index(df)

  ranker = GraphRanker()
  ranker.build_co_review_graph(df)
  pr_scores = ranker.compute_pagerank()

  recommender = RecommenderEngine(df)

  return df, idx, pr_scores, ranker.graph, recommender


@st.cache_resource
def load_system():
  df = pd.read_csv("data/sample_amazon_reviews1.csv").dropna(
      subset=["product_id", "review_body", "customer_id"]
  )
  return build_system(df)


try:
  if "runtime_system" not in st.session_state:
    st.session_state.runtime_system = load_system()
  df, idx, pr_scores, graph, recommender = st.session_state.runtime_system
except Exception as e:
  st.error(
      f"Error initializing system: {e}. Ensure an Amazon reviews CSV is present in `data/`."
  )
  st.stop()

navigation = st.sidebar.radio(
    "Select Module",
    [
        "0. Dashboard & Index Management",
        "1. Ingestion & Text Mining ",
        "2. Web Search & Graph Ranking",
        "3. Recommenders & IR Evaluation ",
    ],
)

if navigation == "0. Dashboard & Index Management":
  st.header("Dashboard & Index Management")
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Review records", f"{len(df):,}")
  c2.metric("Indexed products", f"{len(idx.corpus):,}")
  c3.metric("Vocabulary terms", f"{len(idx.index):,}")
  c4.metric("Graph edges", f"{graph.number_of_edges():,}")
  st.subheader("Index status")
  st.write({"average_document_length": round(idx.avg_doc_len, 2),
            "graph_nodes": graph.number_of_nodes(),
            "pagerank_nodes": len(pr_scores),
            "metadata_fields": ["product_id", "product_title", "star_rating", "product_category"]})
  if st.button("Rebuild index and graph"):
    st.session_state.runtime_system = load_system()
    st.rerun()
  st.info("The index is built from the supplied dataset and is rebuilt through this Streamlit interface.")

# ------------------------------------------------------------------------------
# MODULE 1: 
# ------------------------------------------------------------------------------
elif navigation == "1. Ingestion & Text Mining":
  st.header("Module 1: Data Ingestion, Crawling & Text Preprocessing")
  tab1, tab2 = st.tabs(
      ["Web Crawling Interface", "Text Mining & Corpus Analytics"]
  )

  with tab1:
    st.subheader("Configurable Web Crawler (Section B)")
    seed_urls = st.text_area(
        "Enter seed URLs (one URL per line):",
        "https://en.wikipedia.org/wiki/Information_retrieval"
    )
    col1, col2 = st.columns(2)
    depth = col1.slider("Crawling Depth", 1, 3, 1)
    max_pages = col2.slider("Max Pages to Fetch", 5, 30, 10)

    if st.button("Start Web Crawler"):
      with st.spinner("Crawling web pages..."):
        crawler = SimpleCrawler(max_depth=depth, max_pages=max_pages)
        crawled_results = crawler.crawl(seed_urls.splitlines())
        st.session_state.crawled_results = crawled_results

    if st.session_state.get("crawled_results"):
        crawled_results = st.session_state.crawled_results
        st.success(f"Crawled {len(crawled_results)} unique documents!")
        crawl_table = [
            {
                "Title": item["metadata"]["title"],
                "URL": item["metadata"]["url"],
                "Depth": item["metadata"]["depth"],
                "Content Length": len(item["content"]),
            }
            for item in crawled_results
        ]
        st.dataframe(pd.DataFrame(crawl_table), use_container_width=True)
        if st.button("Add crawled documents to indexed collection"):
          crawl_df = pd.DataFrame([
              {
                  "product_id": item["metadata"]["url"],
                  "customer_id": f"crawl:{item['metadata']['url']}",
                  "product_title": item["metadata"]["title"],
                  "review_body": item["content"],
                  "star_rating": 0,
                  "product_category": item["metadata"]["domain"],
              }
              for item in crawled_results
          ])
          merged_df = pd.concat([df, crawl_df], ignore_index=True)
          st.session_state.runtime_system = build_system(merged_df)
          st.success("Crawled documents were added to the indexed collection.")
          st.rerun()

  with tab2:
    st.subheader("Amazon Corpus Preprocessing & Feature Mining (Section C)")
    miner = TextMiner()
    use_lemma = st.checkbox("Enable Lemmatization", value=True)

    with st.spinner("Mining text features..."):
      processed_corpus = df["review_body"].apply(
          lambda x: miner.preprocess(x, use_lemmatization=use_lemma)
      )

    col_a, col_b = st.columns(2)
    with col_a:
      st.write("**Top 10 Bi-Grams**")
      df_ngrams = miner.get_top_ngrams(processed_corpus, n=2, top_k=10)
      fig_ngram = px.bar(df_ngrams, x="frequency", y="ngram", orientation="h")
      st.plotly_chart(fig_ngram, use_container_width=True)

    with col_b:
      st.write("**Top 10 TF-IDF Features**")
      df_tfidf = miner.extract_tfidf_features(
          processed_corpus, max_features=10
      )
      fig_tfidf = px.bar(df_tfidf, x="mean_tfidf", y="feature", orientation="h")
      st.plotly_chart(fig_tfidf, use_container_width=True)

    st.write("**Corpus Word Cloud**")
    all_text = " ".join(processed_corpus)
    wordcloud = WordCloud(
        width=800, height=300, background_color="white"
    ).generate(all_text)
    st.image(wordcloud.to_array(), use_container_width=True)

    st.subheader("Document profiling and classification")
    profile = miner.document_profile(df)
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Mean words/document", f"{profile['word_count'].mean():.1f}")
    pc2.metric("Mean characters/document", f"{profile['char_count'].mean():.1f}")
    pc3.metric("Categories", int(df["product_category"].nunique()))
    st.dataframe(df["product_category"].value_counts().rename_axis("category").reset_index(name="documents"), use_container_width=True)
    class_result = miner.classify_documents(df)
    st.metric("Category-classification accuracy", f"{class_result['accuracy']:.3f}")

    st.subheader("Comparative preprocessing and feature extraction")
    comparison = []
    for label, lemma in [("Without lemmatization", False), ("With lemmatization", True)]:
      processed = df["review_body"].apply(lambda x: miner.preprocess(x, use_lemmatization=lemma))
      comparison.append({"strategy": label, "mean_tokens": round(processed.str.split().str.len().mean(), 2),
                         "unique_terms": len(set(" ".join(processed).split()))})
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)

    feature_comparison = []
    for label, vectorizer in [
        ("Bag-of-Words", CountVectorizer(stop_words="english", max_features=1000)),
        ("TF-IDF", TfidfVectorizer(stop_words="english", max_features=1000)),
    ]:
      matrix = vectorizer.fit_transform(processed_corpus)
      feature_comparison.append({"feature_strategy": label,
                                 "feature_count": len(vectorizer.get_feature_names_out()),
                                 "nonzero_values": int(matrix.nnz)})
    st.dataframe(pd.DataFrame(feature_comparison), use_container_width=True)
 
# ------------------------------------------------------------------------------
# MODULE 2:
# ------------------------------------------------------------------------------
elif navigation == "2. Web Search & Graph Ranking":
  st.header("Module 2: Search Engine & Link-Graph Ranking")
  st.sidebar.header("Ranking Controls")
  alpha = st.sidebar.slider(
      "BM25 vs PageRank Weight (alpha)",
      0.0,
      1.0,
      0.7,
      help="1.0 = Pure BM25, 0.0 = Pure PageRank",
  )
  top_k = st.sidebar.slider("Top-K Search Hits", 5, 20, 10)

  query = st.text_input("Enter product search query:", "camera lens")

  if query:
    search_started = time.perf_counter()
    raw_results = idx.search_bm25(query, top_k=top_k * 2)
    if not raw_results:
      st.warning("No matching products found.")
    else:
      max_bm25 = max(score for _, score in raw_results) or 1.0
      max_pr = max(pr_scores.values()) if pr_scores else 1.0

      combined_results = []
      for item, bm25_score in raw_results:
        pid = item["product_id"]
        norm_bm25 = bm25_score / max_bm25
        pr = pr_scores.get(pid, 0.0)
        norm_pr = pr / max_pr
        final_score = (alpha * norm_bm25) + ((1 - alpha) * norm_pr)

        combined_results.append({
            "Product Title": item["product_title"],
            "Product ID": pid,
            "Star Rating": round(float(item.get("star_rating", 0)), 2),
            "Category": item.get("product_category", ""),
            "Combined Score": round(final_score, 4),
            "BM25 Score": round(bm25_score, 4),
            "PageRank": round(pr, 6),
        })

      combined_results = sorted(
          combined_results, key=lambda x: x["Combined Score"], reverse=True
      )[:top_k]

      st.subheader("Ranked Search Results (Section D)")
      st.dataframe(pd.DataFrame(combined_results), use_container_width=True)
      rank_plot = pd.DataFrame(combined_results).head(top_k).sort_values("Combined Score")
      st.plotly_chart(px.bar(rank_plot, x="Combined Score", y="Product Title", orientation="h",
                             title="PageRank-aware ranking: final ranked scores"), use_container_width=True)

      col1, col2, col3 = st.columns(3)
      col1.metric("Graph Nodes (Products)", graph.number_of_nodes())
      col2.metric("Graph Edges (Co-Reviews)", graph.number_of_edges())
      col3.metric(
          "Top PageRank Score", f"{combined_results[0]['PageRank']:.6f}"
      )
      st.metric("Search time (seconds)", f"{time.perf_counter() - search_started:.4f}")

# ------------------------------------------------------------------------------
# MODULE 3:
# ------------------------------------------------------------------------------
else:
  st.header("Module 3: Recommender System & IR Evaluation")

  rec_tab, eval_tab = st.tabs(
      ["Recommender Panel (Section E)", "Evaluation Analytics (Section F)"]
  )

  with rec_tab:
    st.subheader("Product Recommender Engine")
    mode = st.radio(
        "Select Recommendation Mode:",
        ["Content-Based (Item-Item)", "Collaborative Filtering (User-Item)"],
    )

    if mode == "Content-Based (Item-Item)":
      sample_pid = st.selectbox(
          "Select Target Product ID:", recommender.product_ids[:15]
      )
      if st.button("Generate Recommendations"):
        recs = recommender.get_content_recommendations(sample_pid, top_k=5)
        st.write(
            "**Top Recommended Items based on Description & Similarity:**"
        )
        st.dataframe(pd.DataFrame(recs), use_container_width=True)

    else:
      sample_user = st.selectbox(
          "Select Target User ID:", recommender.user_ids[:15]
      )
      if st.button("Generate Recommendations"):
        recs = recommender.get_collaborative_recommendations(
            sample_user, top_k=5
        )
        st.write(
            "**Top Recommended Items based on User Ratings (Matrix"
            " Factorization):**"
        )
        st.dataframe(pd.DataFrame(recs), use_container_width=True)

  with eval_tab:
    st.subheader("IR System Evaluation Metrics Dashboard")
    eval_queries = st.text_area(
        "Evaluation Queries (one query per line):",
        "camera lens\ncamera\nbattery",
        key="eval_q",
    )
    k_val = st.slider("Evaluation K Depth", 3, 10, 5)

    if eval_queries.strip():
      evaluation_started = time.perf_counter()
      query_lines = [q.strip() for q in eval_queries.splitlines() if q.strip()]
      bm25_inputs, pagerank_inputs = [], []
      for eval_query in query_lines:
        search_hits = idx.search_bm25(eval_query, top_k=15)
        retrieved_pids = [item["product_id"] for item, _ in search_hits]
        query_tokens = [t for t in re.findall(r"\w+", eval_query.lower()) if len(t) > 2]
        pseudo_ground_truth = set(df[df["product_title"].fillna("").str.lower().apply(
            lambda title: all(token in title for token in query_tokens)
        )]["product_id"].unique())
        bm25_inputs.append((eval_query, retrieved_pids, pseudo_ground_truth))

        max_bm25_eval = max((score for _, score in search_hits), default=1.0) or 1.0
        max_pr_eval = max(pr_scores.values(), default=1.0) or 1.0
        reranked = sorted(
            [(item["product_id"], 0.7 * score / max_bm25_eval +
              0.3 * pr_scores.get(item["product_id"], 0.0) / max_pr_eval)
             for item, score in search_hits], key=lambda x: x[1], reverse=True
        )
        pagerank_inputs.append((eval_query, [pid for pid, _ in reranked], pseudo_ground_truth))

      bm25_detail, bm25_mean = IREvaluator.evaluate_queries(bm25_inputs, k_val)
      pagerank_detail, pagerank_mean = IREvaluator.evaluate_queries(pagerank_inputs, k_val)
      comparison_metrics = pd.DataFrame([
          {"ranking": "BM25", **bm25_mean.iloc[0].drop("Query").to_dict()},
          {"ranking": "BM25 + PageRank", **pagerank_mean.iloc[0].drop("Query").to_dict()},
      ])
      st.write(f"**Evaluation Results across {len(query_lines)} queries (K={k_val})**")
      st.caption("Relevance is defined transparently as products whose titles contain all query terms. MAP is the mean of per-query average precision values.")
      st.subheader("Per-query BM25 results")
      st.dataframe(bm25_detail, use_container_width=True)
      st.subheader("Per-query BM25 + PageRank results")
      st.dataframe(pagerank_detail, use_container_width=True)
      st.subheader("Comparative ranking analysis")
      st.dataframe(comparison_metrics, use_container_width=True)
      st.plotly_chart(px.bar(comparison_metrics, x="ranking", y=f"NDCG@{k_val}",
                             title="NDCG comparison: BM25 versus PageRank-aware ranking"), use_container_width=True)
      st.metric("Evaluation time (seconds)", f"{time.perf_counter() - evaluation_started:.4f}")

      fig_metrics = px.bar(
          x=list(comparison_metrics.columns[1:]),
          y=list(comparison_metrics.iloc[0, 1:]),
          labels={"x": "Metric", "y": "Score"},
          title=f"Retrieval Metrics Performance (K={k_val})",
      )
      st.plotly_chart(fig_metrics, use_container_width=True)
