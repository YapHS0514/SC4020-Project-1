import pandas as pd
import nltk
import cosine_similarity
import jaccard_similarity 

nltk.download('punkt')
nltk.download('punkt_tab')  # In case punkt alone doesn't work

path = "data/News Article Dataset.csv"

df = pd.read_csv(path)
article_texts = df["Article text"]
print(article_texts)

# Fill NaN values with empty strings
df['Article text'] = df['Article text'].fillna('')

# Tokenize sentences for each article text
df['Sentences'] = df['Article text'].apply(lambda x: nltk.sent_tokenize(x))

choice = input("Choose similarity metric (1 for Cosine, 2 for Jaccard): ")

target_sentence = "Lack of lorry drivers globally."

# List to store all sentences and their similarity scores globally
all_similarity_scores = []

# Iterate through each row in the DataFrame
for idx, row in df.iterrows():

    if idx >= 10:  # Stop after 10 rows
        break

    print(f"\nProcessing Article {idx + 1}")
    
    # Iterate through the list of sentences in the 'Sentences' column for the current row
    for sentence in row['Sentences']:

        if choice == "1":
            similarity_score = cosine_similarity.compute_cosine_similarity(target_sentence, sentence)

        elif choice == "2":
            similarity_score = jaccard_similarity.jaccard_similarity(target_sentence, sentence)

        # Append the sentence and its similarity score to the global list
        all_similarity_scores.append((idx, sentence, similarity_score))


# Sort all sentences by similarity score in descending order and get the top 5
top_5_sentences = sorted(all_similarity_scores, key=lambda x: x[2], reverse=True)[:5]

# Print the top 5 most similar sentences globally
for idx, sentence, score in top_5_sentences:
    print(f"\nSentence in Article {idx + 1}: {sentence}")
    print(f"Similarity score: {score:.4f}")
    print("-" * 50)