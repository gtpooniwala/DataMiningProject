from openai import OpenAI

def get_word_lists():
    common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"]
    nouns = ["cat", "dog", "house", "car", "tree", "book", "phone", "computer", "city", "ocean"]
    verbs = ["run", "jump", "eat", "sleep", "write", "read", "swim", "dance", "sing", "think"]
    adjectives = ["happy", "sad", "big", "small", "fast", "slow", "hot", "cold", "new", "old"]
    return common_words, nouns, verbs, adjectives

def generate_embeddings(words):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=words
    )
    return [item.embedding for item in response.data]
