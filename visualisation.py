import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_features(x, y, words, colors, feature_x, feature_y, title, filename):
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(x, y, c=colors, alpha=0.6)

    texts = []
    for i, word in enumerate(words):
        texts.append(plt.text(x[i], y[i], word, fontsize=9))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    plt.title(title)
    plt.xlabel(f"Feature {feature_x}")
    plt.ylabel(f"Feature {feature_y}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_clusters(embeddings_2d, labels, words, title, filename):
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    
    texts = []
    for i, word in enumerate(words):
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=9))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))
    
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_dendrogram(model, words):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=words, leaf_rotation=90., leaf_font_size=8.)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig("dendrogram.png")
    plt.close()
