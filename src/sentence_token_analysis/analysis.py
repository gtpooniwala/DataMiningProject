import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.embeddings import generate_sentence_embeddings

def perform_sentence_analysis():
    # Sample sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fox is a small carnivorous mammal with a bushy tail.",
        "The dog barked loudly at the passing car.",
        "I love reading books in my free time.",
        "Books are a great source of knowledge and entertainment."
    ]

    # Generate sentence embeddings
    sentence_embeddings = generate_sentence_embeddings(sentences)

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(sentence_embeddings)

    # Print sentence pairs and their similarities
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"Sentence 1: {sentences[i]}")
            print(f"Sentence 2: {sentences[j]}")
            print(f"Similarity: {similarities[i, j]:.2f}")
            print()

# Assuming you have implemented the generate_sentence_embeddings function
# in src/utils/embeddings.py

if __name__ == "__main__":
    perform_sentence_analysis()
