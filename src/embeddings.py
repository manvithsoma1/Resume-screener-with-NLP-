from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # using a lightweight model for fast local embeddings
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """Generates sentence embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True)

    def reduce_dimensions_pca(self, embeddings, n_components=2):
        """Reduces embedding dimensionality for visualization using PCA."""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)
        
    def reduce_dimensions_tsne(self, embeddings, n_components=2, perplexity=30):
        """Reduces embedding dimensionality for visualization using t-SNE."""
        # Handle small datasets gracefully
        n_samples = len(embeddings)
        if n_samples < perplexity:
            perplexity = max(1, n_samples - 1)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(embeddings)

if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    sentences = ["I am a software engineer.", "Machine learning is my passion.", "I love coding in Python."]
    embs = embedder.generate_embeddings(sentences)
    print("Embedding shape:", embs.shape)
    pca_embs = embedder.reduce_dimensions_pca(embs)
    print("PCA shape:", pca_embs.shape)
