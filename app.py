import streamlit as st
import pickle
import os
import train

st.title("🎬 Movie Recommendation System")

# Step 1: Ensure pkl files exist
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    st.write("⏳ Preparing data... please wait...")
    train.main()

# Step 2: Load data AFTER creation
@st.cache_resource
def load_data():
    movies = pickle.load(open('movies.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))
    return movies, similarity

movies, similarity = load_data()

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# UI
selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    for movie in recommendations:
        st.write(movie)