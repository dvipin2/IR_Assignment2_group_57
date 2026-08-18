import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# ==========================================
# SECTION E: RECOMMENDER SYSTEM ENGINE
# ==========================================
class AmazonRecommender:
    def __init__(self, df):
        """
        Expects a cleaned DataFrame containing:
        - customer_id
        - product_id
        - product_title
        - star_rating
        - review_body (or cleaned text)
        """
        self.df = df
        
        # Unique products metadata mapping
        self.product_meta = (
            df[['product_id', 'product_title']]
            .drop_duplicates(subset=['product_id'])
            .set_index('product_id')
        )

    def content_based_recommendation(self, product_id, top_k=5):
        """
        Recommends products similar to a target product using TF-IDF on review texts/titles.
        """
        if product_id not in self.product_meta.index:
            return pd.DataFrame()

        # Aggregate review text by product
        product_texts = (
            self.df.groupby('product_id')['review_body']
            .apply(lambda x: ' '.join(x.astype(str)))
            .reset_index()
        )
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(product_texts['review_body'])
        
        # Get target index
        target_idx = product_texts[product_texts['product_id'] == product_id].index[0]
        
        # Compute cosine similarity
        sim_scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
        
        # Rank top K (excluding the product itself)
        top_indices = sim_scores.argsort()[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            p_id = product_texts.iloc[idx]['product_id']
            score = float(sim_scores[idx])
            title = self.product_meta.loc[p_id, 'product_title']
            results.append({'product_id': p_id, 'product_title': title, 'similarity_score': round(score, 4)})
            
        return pd.DataFrame(results)

    def collaborative_recommendation(self, customer_id, top_k=5, n_components=20):
        """
        User-Item Matrix Factorization using SVD for collaborative filtering.
        """
        if customer_id not in self.df['customer_id'].values:
            return pd.DataFrame()

        # Create pivot table (rows: users, cols: items)
        user_item_matrix = self.df.pivot_table(
            index='customer_id', 
            columns='product_id', 
            values='star_rating', 
            aggfunc='mean'
        ).fillna(0)
        
        if customer_id not in user_item_matrix.index:
            return pd.DataFrame()

        # Matrix Factorization using SVD
        svd = TruncatedSVD(n_components=min(n_components, user_item_matrix.shape[1]-1), random_state=42)
        latent_matrix = svd.fit_transform(user_item_matrix)
        reconstructed_matrix = np.dot(latent_matrix, svd.components_)
        
        pred_df = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
        
        # Filter products already rated by user
        user_ratings = user_item_matrix.loc[customer_id]
        unrated_products = user_ratings[user_ratings == 0].index
        
        # Predict scores for unrated products
        predictions = pred_df.loc[customer_id, unrated_products].sort_values(ascending=False).head(top_k)
        
        results = []
        for p_id, score in predictions.items():
            title = self.product_meta.loc[p_id, 'product_title']
            results.append({'product_id': p_id, 'product_title': title, 'predicted_rating': round(score, 4)})
            
        return pd.DataFrame(results)

    def hybrid_recommendation(self, customer_id, product_id, top_k=5, alpha=0.5):
        """
        Combines content-based scores and collaborative ratings.
        """
        cb_df = self.content_based_recommendation(product_id, top_k=top_k*2)
        cf_df = self.collaborative_recommendation(customer_id, top_k=top_k*2)
        
        if cb_df.empty or cf_df.empty:
            return cb_df if not cb_df.empty else cf_df
            
        # Normalize scores between 0 and 1 for combination
        cb_df['score_norm'] = (cb_df['similarity_score'] - cb_df['similarity_score'].min()) / (cb_df['similarity_score'].max() - cb_df['similarity_score'].min() + 1e-6)
        cf_df['score_norm'] = (cf_df['predicted_rating'] - cf_df['predicted_rating'].min()) / (cf_df['predicted_rating'].max() - cf_df['predicted_rating'].min() + 1e-6)
        
        merged = pd.merge(cb_df, cf_df, on=['product_id', 'product_title'], how='outer', suffixes=('_cb', '_cf')).fillna(0)
        merged['hybrid_score'] = alpha * merged['score_norm_cb'] + (1 - alpha) * merged['score_norm_cf']
        
        results = merged.sort_values(by='hybrid_score', ascending=False).head(top_k)
        return results[['product_id', 'product_title', 'hybrid_score']]


