import numpy as np
from src.utils.embeddings import get_word_lists, generate_embeddings
from src.utils.dimensionality_reduction import perform_tsne, perform_pca
from src.utils.clustering import perform_kmeans, perform_hierarchical_clustering, perform_hierarchical_clustering_for_dendrogram
from src.utils.visualisation import plot_features, plot_clusters, plot_dendrogram
from src.word_token_analysis.analysis import perform_word_token_analysis
from src.sentence_token_analysis.analysis import perform_sentence_analysis
# import nltk

def main():
    # nltk.download('all')

    # perform_word_token_analysis()
    perform_sentence_analysis()
    # Add other analysis functions as you develop them


if __name__ == "__main__":
    main()
