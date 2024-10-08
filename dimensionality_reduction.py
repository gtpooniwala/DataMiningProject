from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def perform_tsne(embeddings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(embeddings)

def perform_pca(embeddings, n_components=4):
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    return embeddings_pca, pca.explained_variance_ratio_
