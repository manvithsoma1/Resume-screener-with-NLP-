from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os

class FeatureEngineer:
    def __init__(self, max_features=3000):
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        
    def fit_transform_bow(self, text_series):
        """Fits and transforms text using Bag of Words."""
        return self.bow_vectorizer.fit_transform(text_series)
        
    def transform_bow(self, text_series):
        return self.bow_vectorizer.transform(text_series)

    def fit_transform_tfidf(self, text_series):
        """Fits and transforms text using TF-IDF."""
        return self.tfidf_vectorizer.fit_transform(text_series)
        
    def transform_tfidf(self, text_series):
        return self.tfidf_vectorizer.transform(text_series)

    def get_tfidf_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names_out()

    def get_bow_feature_names(self):
        return self.bow_vectorizer.get_feature_names_out()

    def save_models(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.bow_vectorizer, f)
        with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

    def load_models(self, path='models/'):
        with open(os.path.join(path, 'bow_vectorizer.pkl'), 'rb') as f:
            self.bow_vectorizer = pickle.load(f)
        with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

if __name__ == "__main__":
    fe = FeatureEngineer()
    corpus = ["I love machine learning", "Machine learning is awesome", "NLP is a subfield of AI"]
    tfidf_matrix = fe.fit_transform_tfidf(corpus)
    print("TF-IDF features:", fe.get_tfidf_feature_names())
    print("Matrix shape:", tfidf_matrix.shape)
