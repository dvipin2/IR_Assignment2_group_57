import pandas as pd
import streamlit as st
import math
import re
from collections import defaultdict
import networkx as nx
from collections import defaultdict
import hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from plotly import express as px
from wordcloud import WordCloud


def read_csv():
    """this data is already cleaned & mixed 
    from the dataset downloaded from the kaggle.
    This dataset contains camera and electronics for better 
    diversity. you can always run the notebook 
    and mix the data if you like. please save it in 
    this data directory only."""
    return pd.read_csv("data/sample_amazon_reviews.csv")


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.avg_doc_len = 0
        self.corpus = {}

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def build_index(self, df, id_col='product_id', text_col='review_body'):
        total_len = 0
        for idx, row in df.iterrows():
            doc_id = row[id_col]
            tokens = self._tokenize(str(row[text_col]))
            
            self.corpus[doc_id] = row.to_dict()
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)

            # Store term frequencies
            tf_counts = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1
            
            for token, count in tf_counts.items():
                self.index[token].append((doc_id, count))
                
        self.avg_doc_len = total_len / max(len(df), 1)

    def search_bm25(self, query, k1=1.5, b=0.75, top_k=10):
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

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.corpus[doc_id], score) for doc_id, score in sorted_docs]



class GraphRanker:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_co_review_graph(self, df):
        """Creates directed product edges based on shared user reviews"""
        user_to_products = defaultdict(set)
        for _, row in df.iterrows():
            user_to_products[row['customer_id']].add(row['product_id'])

        # Edge from Product A -> Product B if same user reviewed both
        for products in user_to_products.values():
            prod_list = list(products)
            for i in range(len(prod_list)):
                for j in range(i + 1, len(prod_list)):
                    self.graph.add_edge(prod_list[i], prod_list[j])
                    self.graph.add_edge(prod_list[j], prod_list[i])

    def compute_pagerank(self, alpha=0.85):
        return nx.pagerank(self.graph, alpha=alpha) if len(self.graph) > 0 else {}

    def compute_hits(self, max_iter=100):
        if len(self.graph) == 0:
            return {}, {}
        hubs, authorities = nx.hits(self.graph, max_iter=max_iter)
        return hubs, authorities



class SimpleCrawler:

    def __init__(self, max_depth=2, max_pages=20):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.doc_hashes = set()
        self.crawled_data = []

    def _get_hash(self, content):
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def crawl(self, seed_urls):
        queue = [(url, 0) for url in seed_urls]

        while queue and len(self.crawled_data) < self.max_pages:
            url, depth = queue.pop(0)

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

                # Extract raw text & check for document duplicates
                text_content = soup.get_text(separator=" ", strip=True)
                doc_hash = self._get_hash(text_content)

                if doc_hash in self.doc_hashes:
                    continue  # Skip duplicate document content
                self.doc_hashes.add(doc_hash)

                # Store metadata separately from raw document text
                title = soup.title.string if soup.title else url
                metadata = {
                    "url": url,
                    "title": title,
                    "depth": depth,
                    "domain": urlparse(url).netloc,
                }

                self.crawled_data.append({
                    "metadata": metadata,
                    "content": text_content,
                })

                # Extract sub-links for deeper crawling
                if depth < self.max_depth:
                    for link in soup.find_all("a", href=True):
                        next_url = urljoin(url, link["href"])
                        if (
                            next_url.startswith("http")
                            and next_url not in self.visited_urls
                        ):
                            queue.append((next_url, depth + 1))

            except Exception:
                continue

        return self.crawled_data


# Ensure NLTK resources are available
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class TextMiner:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text, use_lemmatization=True):
        if not isinstance(text, str):
            return ""
        # Lowercase and remove special characters/punctuation
        text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = text.split()

        # Stop-word removal
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]

        # Optional Lemmatization
        if use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]

        return " ".join(tokens)

    def get_top_ngrams(self, corpus, n=2, top_k=15):
        vec = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(
            corpus
        )
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
        return pd.DataFrame({
            "feature": feature_names,
            "mean_tfidf": mean_tfidf,
        }).sort_values(by="mean_tfidf", ascending=False)

# ------------------------------------------------------------------------------
# 2. STREAMLIT APPLICATION DASHBOARD
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Information Retrieval System", page_icon="🔍", layout="wide"
)

st.title("Unified Information Retrieval System")
st.caption("BITS Pilani WILP - Assignment 2")


# Load dataset and cache indexes
@st.cache_resource
def load_system():
  df = pd.read_csv("data/sample_amazon_reviews.csv").dropna(
      subset=["product_id", "review_body", "customer_id"]
  )

  idx = InvertedIndex()
  idx.build_index(df)

  ranker = GraphRanker()
  ranker.build_co_review_graph(df)
  pr_scores = ranker.compute_pagerank()
  hubs, auth_scores = ranker.compute_hits()

  return df, idx, pr_scores, auth_scores, ranker.graph


try:
  df, idx, pr_scores, auth_scores, graph = load_system()
except Exception as e:
  st.error(
      f"Error loading data: {e}. Please place `sample_amazon_reviews.csv` in"
      " `data/` folder."
  )
  st.stop()

# Sidebar Navigation
navigation = st.sidebar.radio(
    "Select System Module",
    [
        "1. Ingestion & Text Mining (Person 1)",
        "2. Web Search & PageRank Engine (Person 2)",
        "3. Recommenders & Evaluation (Person 3 Placeholder)",
    ],
)

# ------------------------------------------------------------------------------
# MODULE 1: INGESTION, CRAWLING & TEXT MINING
# ------------------------------------------------------------------------------
if navigation == "1. Ingestion & Text Mining (Person 1)":
  st.header("Module 1: Data Ingestion, Crawling & Text Preprocessing")

  tab1, tab2 = st.tabs(
      ["Web Crawling Interface", "Text Mining & Corpus Analytics"]
  )

  with tab1:
    st.subheader("Configurable Web Crawler (Section B)")
    seed_url = st.text_input(
        "Enter Seed URL:", "https://en.wikipedia.org/wiki/Information_retrieval"
    )
    col1, col2 = st.columns(2)
    depth = col1.slider("Crawling Depth", 1, 3, 1)
    max_pages = col2.slider("Max Pages to Fetch", 5, 30, 10)

    if st.button("Start Web Crawler"):
      with st.spinner("Crawling web pages..."):
        crawler = SimpleCrawler(max_depth=depth, max_pages=max_pages)
        crawled_results = crawler.crawl([seed_url])
        st.success(f"Crawled {len(crawled_results)} unique documents!")

        crawl_table = []
        for item in crawled_results:
          meta = item["metadata"]
          crawl_table.append({
              "Title": meta["title"],
              "URL": meta["url"],
              "Depth": meta["depth"],
              "Content Length": len(item["content"]),
          })
        st.dataframe(pd.DataFrame(crawl_table), use_container_width=True)

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

# ------------------------------------------------------------------------------
# MODULE 2: SEARCH ENGINE & GRAPH RANKING
# ------------------------------------------------------------------------------
elif navigation == "2. Web Search & PageRank Engine (Person 2)":
  st.header("Module 2: Search Engine & Link-Graph Ranking")

  st.sidebar.header("Ranking Configuration")
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
        auth = auth_scores.get(pid, 0.0)

        final_score = (alpha * norm_bm25) + ((1 - alpha) * norm_pr)

        combined_results.append({
            "Product Title": item["product_title"],
            "Product ID": pid,
            "Combined Score": round(final_score, 4),
            "BM25 Score": round(bm25_score, 4),
            "PageRank": round(pr, 6),
            "HITS Authority": round(auth, 6),
        })

      combined_results = sorted(
          combined_results, key=lambda x: x["Combined Score"], reverse=True
      )[:top_k]

      st.subheader("Ranked Search Results (Section D)")
      st.dataframe(pd.DataFrame(combined_results), use_container_width=True)

      col1, col2, col3 = st.columns(3)
      col1.metric("Graph Nodes (Products)", graph.number_of_nodes())
      col2.metric("Graph Edges (Co-Reviews)", graph.number_of_edges())
      col3.metric(
          "Top PageRank Score", f"{combined_results[0]['PageRank']:.6f}"
      )

# ------------------------------------------------------------------------------
# MODULE 3: RECOMMENDERS & EVALUATION
# ------------------------------------------------------------------------------
else:
  st.header("Module 3: Recommender System & IR Evaluation Dashboard")
  st.info(
      "This section is allocated to Person 3 for Section E (Content-Based /"
      " Collaborative Recommender) and Section F (Precision, Recall, MAP,"
      " NDCG Metrics)."
  )