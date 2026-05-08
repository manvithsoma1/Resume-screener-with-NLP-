from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityMatcher:
    def __init__(self):
        pass

    def calculate_cosine_similarity(self, source_vector, target_vectors):
        """Calculates cosine similarity between a source vector and a list of target vectors."""
        # Reshape source if it's 1D
        if len(source_vector.shape) == 1:
            source_vector = source_vector.reshape(1, -1)
            
        similarities = cosine_similarity(source_vector, target_vectors)
        return similarities[0] # Returns a 1D array of scores

    def rank_items(self, source_vector, target_vectors, item_metadata, top_n=5):
        """Returns the top N items based on cosine similarity."""
        scores = self.calculate_cosine_similarity(source_vector, target_vectors)
        
        # Combine scores with metadata
        ranked_results = []
        for idx, score in enumerate(scores):
            meta = item_metadata[idx].copy()
            meta['Similarity_Score'] = round(score, 4)
            ranked_results.append(meta)
            
        # Sort descending by score
        ranked_results.sort(key=lambda x: x['Similarity_Score'], reverse=True)
        return ranked_results[:top_n]

if __name__ == "__main__":
    matcher = SimilarityMatcher()
    src = np.array([1, 0, 1])
    tgts = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0]])
    meta = [{'id': 1}, {'id': 2}, {'id': 3}]
    print(matcher.rank_items(src, tgts, meta))
