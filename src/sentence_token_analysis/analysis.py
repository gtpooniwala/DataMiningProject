import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from src.utils.embeddings import generate_sentence_embeddings, get_sentence_lists
from src.utils.dimensionality_reduction import perform_tsne
from matplotlib import pyplot as plt
import os
import random
from adjustText import adjust_text

def perform_sentence_analysis():
    sentences, labels = get_sentence_lists()

    # Generate sentence embeddings
    sentence_embeddings = generate_sentence_embeddings(sentences)

    # Scale the embeddings
    scaler = StandardScaler()
    sentence_embeddings = scaler.fit_transform(sentence_embeddings)

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(sentence_embeddings)

    num_pairs = len(sentences) // 2
    correct_closest_pairs = 0
    for i in range(num_pairs):
        question_index = 2 * i
        answer_index = 2 * i + 1

        # Calculate similarity with the relevant answer
        relevant_similarity = similarities[question_index, answer_index]
        print(f"Sentence 1: {sentences[question_index]}")
        print(f"Sentence 2: {sentences[answer_index]}")
        print(f"Relevant Similarity: {relevant_similarity:.2f}")
        print()

        # Select a random answer that is not the relevant one
        random_answer_index = answer_index
        while random_answer_index == answer_index:
            random_answer_index = random.choice(range(1, len(sentences), 2))

        # Calculate similarity with the random answer
        random_similarity = similarities[question_index, random_answer_index]
        print(f"Sentence 1: {sentences[question_index]}")
        print(f"Sentence 2: {sentences[random_answer_index]}")
        print(f"Random Similarity: {random_similarity:.2f}")
        print()

        # Find the closest answer based on cosine similarity
        closest_answer_index = np.argmax(similarities[question_index, 1::2]) * 2 + 1
        if closest_answer_index == answer_index:
            correct_closest_pairs += 1

    closest_accuracy = correct_closest_pairs / num_pairs
    print(f"\nAccuracy of finding the closest answer based on cosine similarity: {closest_accuracy:.2f}")

    # Apply t-SNE to reduce dimensionality to 2D
    embeddings_2d = perform_tsne(sentence_embeddings, n_components=2)

    # Assign colors to each question-answer pair
    colors = []
    for i in range(num_pairs):
        color = plt.cm.tab20(i % 20)  # Use a colormap with enough distinct colors
        colors.extend([color, color])

    # Plot the t-SNE results without clustering labels
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
    texts = [plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], label, fontsize=9, color='black', alpha=0.7) for i, label in enumerate(labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))

    plt.title("t-SNE visualization of sentence embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/sentence_token_analysis')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "tsne_sentence_embeddings.png"))
    plt.close()

    # Perform K-means clustering
    n_clusters = num_pairs
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(sentence_embeddings)

    # Check if the clustering algorithm has correctly clustered the question-answer pairs
    correct_kmeans_pairs = 0
    for i in range(num_pairs):
        question_index = 2 * i
        answer_index = 2 * i + 1
        if kmeans_labels[question_index] == kmeans_labels[answer_index]:
            correct_kmeans_pairs += 1
        else:
            print(f"Wrong cluster (K-means): {question_index}")
            print(sentences[question_index])

    kmeans_accuracy = correct_kmeans_pairs / num_pairs
    print(f"\nAccuracy of K-means clustering: {kmeans_accuracy:.2f}")

    # Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(sentence_embeddings)

    # Check if the clustering algorithm has correctly clustered the question-answer pairs
    correct_hierarchical_pairs = 0
    for i in range(num_pairs):
        question_index = 2 * i
        answer_index = 2 * i + 1
        if hierarchical_labels[question_index] == hierarchical_labels[answer_index]:
            correct_hierarchical_pairs += 1
        else:
            print(f"Wrong cluster (Hierarchical): {question_index}")
            print(sentences[question_index])

    hierarchical_accuracy = correct_hierarchical_pairs / num_pairs
    print(f"\nAccuracy of hierarchical clustering: {hierarchical_accuracy:.2f}")

    # Plot the t-SNE results with K-means clustering labels
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        cluster_points = embeddings_2d[kmeans_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    texts = [plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], label, fontsize=9, color='black', alpha=0.7) for i, label in enumerate(labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))

    plt.title("t-SNE visualization of sentence embeddings with K-means clustering")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(results_dir, "kmeans_sentence_clusters.png"), bbox_inches='tight')
    plt.close()

    # Plot the t-SNE results with hierarchical clustering labels
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        cluster_points = embeddings_2d[hierarchical_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    texts = [plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], label, fontsize=9, color='black', alpha=0.7) for i, label in enumerate(labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))

    plt.title("t-SNE visualization of sentence embeddings with hierarchical clustering")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(results_dir, "hierarchical_sentence_clusters.png"), bbox_inches='tight')
    plt.close()

    print(f"\nAll sentence token analysis plots have been saved as PNG files in the '{results_dir}' directory.")

if __name__ == "__main__":
    perform_sentence_analysis()
