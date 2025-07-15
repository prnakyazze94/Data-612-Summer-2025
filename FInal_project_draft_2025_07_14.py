import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------------------
# Load Enriched Movie Data
# ----------------------------------------

file_path = r"C:\Users\pricc\Downloads\movies_enriched_full.csv"
df = pd.read_csv(file_path)

# ----------------------------------------
# Combine Metadata Fields
# ----------------------------------------
def combine_metadata(row):
    return " ".join([
        str(row["tmdb_genres"]) if pd.notnull(row["tmdb_genres"]) else "",
        str(row["keywords"]) if pd.notnull(row["keywords"]) else "",
        str(row["top_3_cast"]) if pd.notnull(row["top_3_cast"]) else "",
        str(row["directors"]) if pd.notnull(row["directors"]) else ""
    ]).lower().replace(",", " ").replace(":", " ").replace("-", " ")

df["metadata"] = df.apply(combine_metadata, axis=1)

# ----------------------------------------
# Build Vectorizers
# ----------------------------------------

# Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df["metadata"])

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df["metadata"])

# ----------------------------------------
# Compute Cosine Similarity
# ----------------------------------------

cosine_sim_count = cosine_similarity(count_matrix)
cosine_sim_tfidf = cosine_similarity(tfidf_matrix)

# ----------------------------------------
# Save Results
# ----------------------------------------

# Save similarity matrices as NumPy arrays
np.save("cosine_sim_count.npy", cosine_sim_count)
np.save("cosine_sim_tfidf.npy", cosine_sim_tfidf)

# Optional: Save similarity matrices as CSVs
pd.DataFrame(cosine_sim_count, index=df["title"], columns=df["title"]).to_csv("cosine_sim_count.csv")
pd.DataFrame(cosine_sim_tfidf, index=df["title"], columns=df["title"]).to_csv("cosine_sim_tfidf.csv")

# ----------------------------------------
# Recommendation Functions
# ----------------------------------------

def recommend_movies(title, similarity_matrix, df, top_n=10):
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_movies = [(df.iloc[i]["title"], score) for i, score in sim_scores]
    return similar_movies

def print_recommendations(title, similarity_matrix, df, top_n=5):
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    print(f"\n Top {top_n} similar movies to '{title}':")
    for i, (movie_idx, score) in enumerate(sim_scores, 1):
        print(f"{i}. {df.iloc[movie_idx]['title']} (Similarity: {score:.4f})")

# ----------------------------------------
# Run Example Recommendations and Print Output
# ----------------------------------------

print(" Models built and similarity matrices saved.\n")

print(" Cosine Similarity Matrix (CountVectorizer) [first 5x5]:")
print(pd.DataFrame(cosine_sim_count, index=df["title"], columns=df["title"]).iloc[:5, :5])

print("\n TF-IDF Recommendations for 'Toy Story (1995)'")
print_recommendations("Toy Story (1995)", cosine_sim_tfidf, df, top_n=5)

print("\n CountVectorizer Recommendations for 'Toy Story (1995)'")
print_recommendations("Toy Story (1995)", cosine_sim_count, df, top_n=5)

print("\n Output files saved:")
print("  - cosine_sim_count.npy")
print("  - cosine_sim_tfidf.npy")
print("  - cosine_sim_count.csv")
print("  - cosine_sim_tfidf.csv")