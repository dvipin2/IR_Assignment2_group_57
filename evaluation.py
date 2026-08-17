# ==========================================
# SECTION F: IR METRICS EVALUATION SYSTEM
# ==========================================
class IREvaluationMetrics:
    @staticmethod
    def precision_at_k(actual_relevant, retrieved_list, k):
        retrieved_at_k = retrieved_list[:k]
        relevant_retrieved = set(retrieved_at_k).intersection(set(actual_relevant))
        return len(relevant_retrieved) / k

    @staticmethod
    def recall_at_k(actual_relevant, retrieved_list, k):
        if not actual_relevant:
            return 0.0
        retrieved_at_k = retrieved_list[:k]
        relevant_retrieved = set(retrieved_at_k).intersection(set(actual_relevant))
        return len(relevant_retrieved) / len(actual_relevant)

    @staticmethod
    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(actual_relevant, retrieved_list):
        for idx, item in enumerate(retrieved_list):
            if item in actual_relevant:
                return 1.0 / (idx + 1)
        return 0.0

    @staticmethod
    def average_precision(actual_relevant, retrieved_list, k=10):
        if not actual_relevant:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for i, item in enumerate(retrieved_list[:k]):
            if item in actual_relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        return sum_precisions / min(len(actual_relevant), k)

    @staticmethod
    def ndcg_at_k(actual_relevant, retrieved_list, k):
        dcg = 0.0
        idcg = 0.0
        for i, item in enumerate(retrieved_list[:k]):
            rel = 1 if item in actual_relevant else 0
            dcg += rel / np.log2(i + 2)
            
        # Calculate Ideal DCG
        ideal_rels = [1] * min(len(actual_relevant), k)
        for i, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0