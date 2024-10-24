from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# def perform_tsne(embeddings, n_components=2):
#     tsne = TSNE(n_components=n_components, random_state=42)
#     return tsne.fit_transform(embeddings)

# from sklearn.manifold import TSNE


def perform_tsne(embeddings, n_components=2, perplexity=30):
    num_embeddings = len(embeddings)
    
    if num_embeddings == 1:
        # If there is only one embedding, return it as is
        return embeddings
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    
    # Check if perplexity is less than the number of embeddings
    if perplexity >= num_embeddings:
        # Adjust perplexity to be less than the number of embeddings
        perplexity = num_embeddings - 2
        if perplexity < 1:
            # If adjusted perplexity is less than 1, set it to a small positive value
            perplexity = 0.1
        print(f"Perplexity adjusted to {perplexity} to match the number of embeddings.")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    
    return tsne.fit_transform(embeddings)

def perform_pca(embeddings, n_components=4):
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    return embeddings_pca, pca.explained_variance_ratio_
