from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def perform_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

def perform_hierarchical_clustering(embeddings, n_clusters=5):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    return hierarchical.fit_predict(embeddings)

def perform_hierarchical_clustering_for_dendrogram(embeddings):
    hierarchical = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    return hierarchical.fit(embeddings)
