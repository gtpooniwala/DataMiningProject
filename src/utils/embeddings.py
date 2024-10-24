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
        input_text = " ".join(texts)
    else:
        # Join words into a single string
        input_text = " ".join(texts)

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input_text
    )

    if is_sentence:
        # Return the single embedding for the combined sentences
        return [response.data[0].embedding]
    else:
        # Return the embeddings for individual words
        return [item.embedding for item in response.data]

def get_sentence_lists():
    sentences = [
        "The cat chased the mouse across the room.",
        "I love reading books on rainy days.",
        "She enjoys painting beautiful landscapes.",
        "The team won the championship game.",
        "Eating healthy foods is important for good health.",
        "The sun was shining brightly in the clear sky.",
        "Learning a new language can be challenging but rewarding.",
        "The concert was an amazing experience.",
        "He always arrives early for his meetings.",
        "Traveling to new places broadens your horizons.",
        "The movie had an unexpected plot twist.",
        "She excelled in her mathematics class.",
        "The scientist made a groundbreaking discovery.",
        "The beach was crowded on the hot summer day.",
        "Exercise is crucial for maintaining a healthy lifestyle.",
        "The company launched a new product line.",
        "He is an avid collector of rare coins.",
        "The book was a bestseller and received critical acclaim.",
        "She enjoys hiking in the mountains.",
        "The city was bustling with activity."
    ]
    return sentences

def generate_sentence_embeddings(sentences):
    return generate_embeddings(sentences, is_sentence=True)
