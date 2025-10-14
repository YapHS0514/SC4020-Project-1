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
    
    print(similarities[0][0])


if __name__ == "__main__":
    target_sentence  ="i love chocolate ice cream on a rainy day"
    text_sentence = "i hate chocolate ice cream on a rainy day"
    compute_cosine_similarity(target_sentence, text_sentence)
