import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.embeddings import generate_sentence_embeddings, get_sentence_lists
from src.utils.dimensionality_reduction import perform_tsne
from matplotlib import pyplot as plt
import os
import random

def perform_sentence_analysis():
    sentences, labels = get_sentence_lists()

    # Generate sentence embeddings
    sentence_embeddings = generate_sentence_embeddings(sentences)

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(sentence_embeddings)

    # # Print sentence pairs and their similarities
    # for i in range(len(sentences)):
    #     for j in range(i + 1, len(sentences)):
    #         print(f"Sentence 1: {sentences[i]}")
    #         print(f"Sentence 2: {sentences[j]}")
    #         print(f"Similarity: {similarities[i, j]:.2f}")
    #         print()

    num_pairs = len(sentences) // 2
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

    # Apply t-SNE to reduce dimensionality to 2D
    embeddings_2d = perform_tsne(sentence_embeddings, n_components=2)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
        plt.text(embeddings_2d[i, 0] + 0.01, embeddings_2d[i, 1] + 0.01, label, fontsize=9)

    plt.title("t-SNE visualization of sentence embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/sentence_token_analysis')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "tsne_sentence_embeddings.png"))
    plt.close()


if __name__ == "__main__":
    perform_sentence_analysis()
