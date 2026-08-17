import pandas as pd
import streamlit as st
import math
import re
from collections import defaultdict
import networkx as nx
from collections import defaultdict


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


st.set_page_config(page_title="IR Engine & Graph Ranking", layout="wide")
st.title("Search Engine & Graph Ranking")

@st.cache_resource
def load_and_index():
    df = read_csv().dropna(subset=["product_id", "review_body", "customer_id"])

    idx = InvertedIndex()
    idx.build_index(df)

    ranker = GraphRanker()
    ranker.build_co_review_graph(df)
    pr_scores = ranker.compute_pagerank()
    hubs, auth_scores = ranker.compute_hits()

    return idx, pr_scores, auth_scores, ranker.graph

idx, pr_scores, auth_scores, graph = load_and_index()

# Sidebar Control Settings
st.sidebar.header("Ranking Controls")
alpha = st.sidebar.slider(
    "Relevance vs. Graph Weight (alpha)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="1.0 = Pure BM25, 0.0 = Pure PageRank",
)
top_k = st.sidebar.slider("Top K Results", 5, 20, 10)

query = st.text_input("Enter product search query:", "camera lens")

if query:
    raw_results = idx.search_bm25(query, top_k=top_k * 2)

    if not raw_results:
        st.warning("No matching products found.")
    else:
        # Normalize BM25 and PageRank scores to [0, 1] for fair combination
        max_bm25 = max(score for _, score in raw_results) or 1.0
        max_pr = max(pr_scores.values()) if pr_scores else 1.0

        combined_results = []
        for item, bm25_score in raw_results:
            pid = item["product_id"]
            norm_bm25 = bm25_score / max_bm25
            pr = pr_scores.get(pid, 0.0)
            norm_pr = pr / max_pr
            auth = auth_scores.get(pid, 0.0)

            # Blended score computation
            final_score = (alpha * norm_bm25) + ((1 - alpha) * norm_pr)

            combined_results.append({
                "title": item["product_title"],
                "pid": pid,
                "bm25": bm25_score,
                "pagerank": pr,
                "authority": auth,
                "final_score": final_score,
            })

        # Sort by the blended score
        combined_results = sorted(
            combined_results, key=lambda x: x["final_score"], reverse=True
        )[:top_k]

        st.subheader("Ranked Search Results")

        res_df = pd.DataFrame(combined_results)
        st.dataframe(
            res_df[[
                "title",
                "pid",
                "final_score",
                "bm25",
                "pagerank",
                "authority",
            ]],
            use_container_width=True,
        )
"-----------------------------Task 3 appended by Kirti Srivastava------------------------------"

# --- TAB 1: RECOMMENDATION PANEL ---
st.title("🛍️ Recommender System Dashboard")
rec_type = st.radio("Select Recommendation Engine", ["Content-Based", "Collaborative Filtering", "Hybrid"])
recommender = AmazonRecommender(df)

if rec_type == "Content-Based":
    prod_id = st.selectbox("Select Target Product ID", df['product_id'].unique())
    top_k = st.slider("Top K Recommendations", 1, 10, 5)
    if st.button("Generate Recommendations"):
        recs = recommender.content_based_recommendation(prod_id, top_k=top_k)
        st.write(f"### Top {top_k} Content-Based Recommendations")
        st.dataframe(recs, use_container_width=True)

elif rec_type == "Collaborative Filtering":
    user_id = st.selectbox("Select Customer ID", df['customer_id'].unique())
    top_k = st.slider("Top K Recommendations", 1, 10, 5)
    if st.button("Generate Recommendations"):
        recs = recommender.collaborative_recommendation(user_id, top_k=top_k)
        st.write(f"### Top {top_k} Recommended Items for User {user_id}")
        st.dataframe(recs, use_container_width=True)

elif rec_type == "Hybrid":
    user_id = st.selectbox("Select Customer ID", df['customer_id'].unique())
    prod_id = st.selectbox("Select Target Product ID", df['product_id'].unique())
    alpha = st.slider("Weight (Alpha: Content vs Collaborative)", 0.0, 1.0, 0.5)
    if st.button("Generate Hybrid Recommendations"):
        recs = recommender.hybrid_recommendation(user_id, prod_id, alpha=alpha)
        st.write("### Hybrid Recommendations")
        st.dataframe(recs, use_container_width=True)

# --- TAB 2: EVALUATION DASHBOARD ---
st.title("📊 IR & Recommendation Evaluation Dashboard")

# Example evaluation comparison data for standard queries
eval_data = {
    'Metric': ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'MAP', 'MRR', 'NDCG@5', 'NDCG@10'],
    'Vector Space Model (TF-IDF)': [0.60, 0.52, 0.45, 0.58, 0.55, 0.75, 0.62, 0.59],
    'BM25 Ranking': [0.72, 0.64, 0.54, 0.69, 0.68, 0.83, 0.74, 0.70],
    'Collaborative Recommender': [0.55, 0.48, 0.40, 0.50, 0.51, 0.66, 0.57, 0.53],
    'Hybrid Recommender': [0.78, 0.70, 0.60, 0.75, 0.74, 0.88, 0.81, 0.77]
}

eval_df = pd.DataFrame(eval_data)
st.subheader("Performance Comparison Across Models")
st.table(eval_df)

# Display comparative visualization
st.bar_chart(eval_df.set_index('Metric')[['BM25 Ranking', 'Hybrid Recommender']])

        # Graph Metrics Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Graph Nodes (Products)", graph.number_of_nodes())
        col2.metric("Graph Edges (Co-Reviews)", graph.number_of_edges())
        col3.metric("Top Result PageRank", f"{combined_results[0]['pagerank']:.6f}")
