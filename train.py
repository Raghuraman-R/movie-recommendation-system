import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def main():
    print("⏳ Training model...")

    # Load dataset
    movies = pd.read_csv('movies.csv')

    # Select required columns
    movies = movies[['title', 'overview']]
    movies.dropna(inplace=True)

    # Text vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['overview']).toarray()

    # Similarity matrix
    similarity = cosine_similarity(vectors)

    # Save files
    pickle.dump(movies, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

    print("✅ Model trained successfully!")

# This ensures it runs when called directly OR from app.py
if __name__ == "__main__":
    main()