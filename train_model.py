import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("medicine_data.csv")

# Convert 'Reason' column into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Reason"])

# Compute similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Save model and vectorizer
with open("medicine_model.pkl", "wb") as file:
    pickle.dump((vectorizer, similarity_matrix), file)

print("Model trained and saved!")

