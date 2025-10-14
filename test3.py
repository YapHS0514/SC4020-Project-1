
# Compute the Jaccard similarity between two sentences.
def jaccard_similarity(sentence1, sentence2):

    # Tokenize the sentences into sets of words
    set1 = set(sentence1.lower().split())  # Tokenize and convert to lowercase
    set2 = set(sentence2.lower().split())

    # Compute intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Return Jaccard similarity (intersection / union)
    return intersection / union if union != 0 else 0

if __name__ == "__main__":
    sentence1 = "i love chocolate ice cream on a rainy day"
    sentence2 = "i hate chocolate ice cream on a rainy day"
    similarity = jaccard_similarity(sentence1, sentence2)
    print(f"Jaccard Similarity: {similarity:.4f}")