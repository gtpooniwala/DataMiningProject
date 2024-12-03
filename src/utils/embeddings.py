# from openai import OpenAI

# def get_word_lists():
#     common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"]
#     nouns = ["cat", "dog", "house", "car", "tree", "book", "phone", "computer", "city", "ocean"]
#     verbs = ["run", "jump", "eat", "sleep", "write", "read", "swim", "dance", "sing", "think"]
#     adjectives = ["happy", "sad", "big", "small", "fast", "slow", "hot", "cold", "new", "old"]
#     return common_words, nouns, verbs, adjectives

# def generate_embeddings(words):
#     client = OpenAI()
#     response = client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=words
#     )
#     return [item.embedding for item in response.data]

from openai import OpenAI
import numpy as np

def get_word_lists():
    common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"]
    nouns = ["cat", "dog", "house", "car", "tree", "book", "phone", "computer", "city", "ocean"]
    verbs = ["run", "jump", "eat", "sleep", "write", "read", "swim", "dance", "sing", "think"]
    adjectives = ["happy", "sad", "big", "small", "fast", "slow", "hot", "cold", "new", "old"]
    return common_words, nouns, verbs, adjectives

def generate_embeddings(texts, is_sentence=False):
    client = OpenAI()
    if is_sentence:
        # Combine sentences into a single string
        input_text = texts  # Use texts directly as a list of sentences
        embeddings = []
        for text in input_text:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
    else:
        # Join words into a single string
        input_text = " ".join(texts)

        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    

def get_sentence_lists():
    sentences = [
        "What is the capital of France?",  # Question
        "The capital of France is Paris.",  # Answer
        "How does photosynthesis work?",  # Question
        "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",  # Answer
        "What is the largest planet in our solar system?",  # Question
        "The largest planet in our solar system is Jupiter.",  # Answer
        "Who wrote 'To Kill a Mockingbird'?",  # Question
        "'To Kill a Mockingbird' was written by Harper Lee.",  # Answer
        "What is the boiling point of water?",  # Question
        "The boiling point of water is 100 degrees Celsius.",  # Answer
        "What is the speed of light?",  # Question
        "The speed of light is approximately 299,792 kilometers per second.",  # Answer
        "Who was the first president of the United States?",  # Question
        "The first president of the United States was George Washington.",  # Answer
        "What is the chemical symbol for gold?",  # Question
        "The chemical symbol for gold is Au.",  # Answer
        "How many continents are there on Earth?",  # Question
        "There are seven continents on Earth.",  # Answer
        "What is the tallest mountain in the world?",  # Question
        "The tallest mountain in the world is Mount Everest.",  # Answer
        "What is the main ingredient in guacamole?",  # Question
        "The main ingredient in guacamole is avocado.",  # Answer
        "What is the freezing point of water?",  # Question
        "The freezing point of water is 0 degrees Celsius.",  # Answer
        "Who developed the theory of relativity?",  # Question
        "The theory of relativity was developed by Albert Einstein.",  # Answer
        "What is the largest ocean on Earth?",  # Question
        "The largest ocean on Earth is the Pacific Ocean.",  # Answer
        "What is the currency of Japan?",  # Question
        "The currency of Japan is the yen.",  # Answer
        "What is the smallest prime number?",  # Question
        "The smallest prime number is 2.",  # Answer
    ]
    labels = [
        "Capital of France (Q)", "Capital of France (A)",
        "Photosynthesis (Q)", "Photosynthesis (A)",
        "Largest planet (Q)", "Largest planet (A)",
        "Author of 'To Kill a Mockingbird' (Q)", "Author of 'To Kill a Mockingbird' (A)",
        "Boiling point of water (Q)", "Boiling point of water (A)",
        "Speed of light (Q)", "Speed of light (A)",
        "First president of the USA (Q)", "First president of the USA (A)",
        "Chemical symbol for gold (Q)", "Chemical symbol for gold (A)",
        "Number of continents (Q)", "Number of continents (A)",
        "Tallest mountain (Q)", "Tallest mountain (A)",
        "Main ingredient in guacamole (Q)", "Main ingredient in guacamole (A)",
        "Freezing point of water (Q)", "Freezing point of water (A)",
        "Theory of relativity (Q)", "Theory of relativity (A)",
        "Largest ocean (Q)", "Largest ocean (A)",
        "Currency of Japan (Q)", "Currency of Japan (A)",
        "Smallest prime number (Q)", "Smallest prime number (A)"
    ]
    return sentences, labels

def generate_sentence_embeddings(sentences):
    return generate_embeddings(sentences, is_sentence=True)
