from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2') 

def compute_cosine_similarity(target_sentence, text_sentence):
    # Encode the target text and the list of texts
    target_embedding = model.encode([target_sentence])
    text_embeddings = model.encode([text_sentence])

    # Compute cosine similarities
    similarities = cosine_similarity(target_embedding, text_embeddings)

    # Print the similarity scores
    #for score in enumerate(similarities[0]):
     #   print(f"Sentence: {text_sentence}")
     #   print(f"Similarity score: {score:.4f}")
     #   print("-" * 50)

    return similarities[0][0]

if __name__ == "__main__":
    target = "The economy is improving steadily."
    text = "(CNN)Right now, there's a shortage of truck drivers in the US and worldwide, exacerbated by the e-commerce boom brought on by the pandemic."
    compute_cosine_similarity(target, text)
