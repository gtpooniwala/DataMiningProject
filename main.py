import numpy as np
from embeddings import get_word_lists, generate_embeddings
from dimensionality_reduction import perform_tsne, perform_pca
from clustering import perform_kmeans, perform_hierarchical_clustering, perform_hierarchical_clustering_for_dendrogram
from visualisation import plot_features, plot_clusters, plot_dendrogram

def main():
    # Get word lists and generate embeddings
    common_words, nouns, verbs, adjectives = get_word_lists()
    all_words = common_words + nouns + verbs + adjectives
    embeddings = generate_embeddings(all_words)
    embeddings_array = np.array(embeddings)

    # Set up colors for each category
    categories = (["common"] * len(common_words) + 
                  ["noun"] * len(nouns) + 
                  ["verb"] * len(verbs) + 
                  ["adjective"] * len(adjectives))
    color_map = {"common": "blue", "noun": "red", "verb": "green", "adjective": "purple"}
    colors = [color_map[cat] for cat in categories]

    # Perform t-SNE
    embeddings_2d = perform_tsne(embeddings_array, n_components=2)
    embeddings_3d = perform_tsne(embeddings_array, n_components=3)

    # Plot t-SNE results
    plot_features(embeddings_2d[:, 0], embeddings_2d[:, 1], all_words, colors, 1, 2,
                  "t-SNE visualization of word embeddings (2D)", "tsne_2d.png")
    plot_features(embeddings_3d[:, 0], embeddings_3d[:, 1], all_words, colors, 1, 2,
                  "t-SNE visualization of word embeddings (3D, features 1 and 2)", "tsne_3d_features_1_2.png")
    plot_features(embeddings_3d[:, 0], embeddings_3d[:, 2], all_words, colors, 1, 3,
                  "t-SNE visualization of word embeddings (3D, features 1 and 3)", "tsne_3d_features_1_3.png")
    plot_features(embeddings_3d[:, 1], embeddings_3d[:, 2], all_words, colors, 2, 3,
                  "t-SNE visualization of word embeddings (3D, features 2 and 3)", "tsne_3d_features_2_3.png")

    # Perform PCA
    embeddings_pca, explained_variance_ratios = perform_pca(embeddings_array)

    # Plot PCA results
    plot_features(embeddings_pca[:, 0], embeddings_pca[:, 1], all_words, colors, 1, 2,
                  "PCA visualization of word embeddings (1st and 2nd components)", "pca_components_1_2.png")
    plot_features(embeddings_pca[:, 2], embeddings_pca[:, 3], all_words, colors, 3, 4,
                  "PCA visualization of word embeddings (3rd and 4th components)", "pca_components_3_4.png")

    # Print explained variance ratios
    print("\nExplained Variance Ratios for PCA components:")
    for i, ratio in enumerate(explained_variance_ratios, 1):
        print(f"Component {i}: {ratio:.4f}")

    # Perform clustering
    n_clusters = 5
    kmeans_labels = perform_kmeans(embeddings_array, n_clusters)
    hierarchical_labels = perform_hierarchical_clustering(embeddings_array, n_clusters)

    # Plot clustering results
    plot_clusters(embeddings_2d, kmeans_labels, all_words, "K-means Clustering", "kmeans_clusters.png")
    plot_clusters(embeddings_2d, hierarchical_labels, all_words, "Hierarchical Clustering", "hierarchical_clusters.png")

    # Print cluster compositions
    print("\nK-means Clusters:")
    for i in range(n_clusters):
        cluster_words = [word for word, label in zip(all_words, kmeans_labels) if label == i]
        print(f"Cluster {i}: {', '.join(cluster_words)}")

    print("\nHierarchical Clusters:")
    for i in range(n_clusters):
        cluster_words = [word for word, label in zip(all_words, hierarchical_labels) if label == i]
        print(f"Cluster {i}: {', '.join(cluster_words)}")

    # Create and plot dendrogram
    hierarchical_model = perform_hierarchical_clustering_for_dendrogram(embeddings_array)
    plot_dendrogram(hierarchical_model, all_words)

    print("\nAll plots have been saved as PNG files.")

if __name__ == "__main__":
    main()
