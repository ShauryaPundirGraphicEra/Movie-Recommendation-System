import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. Load Data + Embeddings
# --------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies.csv")  # your metadata
        embeddings = np.load("all_embeddings.npy")  # your saved embeddings
        if len(df) != embeddings.shape[0]:
            st.error(
                f"Mismatch between movies_df and embeddings length!\n"
                f"Number of movies in full_moviess.csv: {len(df)}\n"
                f"Number of embeddings in all_embeddings.npy: {embeddings.shape[0]}"
            )
            st.write("Please ensure the number of movies matches the number of embeddings. "
                     "You may need to regenerate all_embeddings.npy or filter full_moviess.csv.")
            st.stop()
        return df, embeddings
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

movies_df, embeddings = load_data()

# Initialize SentenceTransformer model
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer("all-mpnet-base-v2")  # Updated to match 768 dimensions
        expected_dim = 768  # Expected dimension for all-mpnet-base-v2
        if embeddings.shape[1] != expected_dim:
            st.error(
                f"Embedding dimension mismatch!\n"
                f"Model 'all-mpnet-base-v2' produces {expected_dim}-dimensional embeddings, "
                f"but all_embeddings.npy has dimension {embeddings.shape[1]}."
            )
            st.write("Please regenerate all_embeddings.npy with 'all-mpnet-base-v2' or use the correct model.")
            st.stop()
        return model
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {str(e)}")
        st.stop()

model = load_model()

# --------------------------
# 2. Helper: Get Recommendations
# --------------------------
def get_recommendations(movie_id, top_k=5):
    try:
        # Find index of selected movie
        idx = movies_df.index[movies_df["id"] == movie_id]
        if len(idx) == 0:
            st.warning(f"Movie ID {movie_id} not found in dataset.")
            return pd.DataFrame()  # Return empty DataFrame if movie not found
        idx = idx[0]

        # Compute cosine similarity
        movie_emb = embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(movie_emb, embeddings)[0]

        # Sort by similarity (skip the movie itself)
        similar_idx = np.argsort(sims)[::-1][1:top_k+1]

        rec_df = movies_df.iloc[similar_idx]
        return rec_df
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()

# --------------------------
# 3. UI: Home Page
# --------------------------
st.title("🎬 Movie Recommendation System")

# TMDb poster base URL (adjust if needed)
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Search bar
st.text_input("Search for a movie (e.g., 'Adventure treasure jungle')", key="search_query")
if st.button("Search"):
    query_text = st.session_state["search_query"]
    if query_text.strip():
        try:
            # Generate embedding for the query
            query_embedding = model.encode([query_text])[0].reshape(1, -1)
            # Compute cosine similarity with all embeddings
            sims = cosine_similarity(query_embedding, embeddings)[0]
            # Get indices of top 10 most similar movies
            top_indices = np.argsort(sims)[::-1][:10]
            # Store results as a DataFrame slice
            st.session_state["search_results"] = movies_df.iloc[top_indices]
            st.rerun()
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
    else:
        st.warning("Please enter a search query.")

# Pagination setup
page_size = 20
page_num = st.session_state.get("page_num", 0)

start_idx = page_num * page_size
end_idx = start_idx + page_size

# Display movies grid
cols = st.columns(3)
for i, (_, row) in enumerate(movies_df.iloc[start_idx:end_idx].iterrows()):
    with cols[i % 3]:
        # Show poster if available
        try:
            if pd.notna(row["poster_path"]) and row["poster_path"] != "":
                st.image(POSTER_BASE_URL + row["poster_path"], use_container_width=True)
            else:
                st.image("default_poster.jpg", use_container_width=True)
        except Exception as e:
            st.write("🚫 Image Load Failed")

        # Movie title as a button
        if st.button(row["title"], key=f"home_{row['id']}_{i}"):
            st.session_state["selected_movie"] = row["id"]
            st.rerun()

# "Load More" button
if end_idx < len(movies_df):
    if st.button("Load More"):
        st.session_state["page_num"] = page_num + 1
        st.rerun()

# --------------------------
# 4. UI: Movie Detail Page
# --------------------------
if "selected_movie" in st.session_state:
    movie_id = st.session_state["selected_movie"]
    # Find the movie
    if movie_id not in movies_df["id"].values:
        st.error(f"Selected movie ID {movie_id} not found!")
    else:
        movie = movies_df[movies_df["id"] == movie_id].iloc[0]

        # Main movie details
        st.header(movie["title"])
        try:
            if pd.notna(movie["poster_path"]) and movie["poster_path"] != "":
                st.image(POSTER_BASE_URL + movie["poster_path"], width=250)
            else:
                st.image("default_poster.jpg", width=250)
        except Exception as e:
            st.write("🚫 Image Load Failed")
        st.write(f"**📅 Release Year:** {movie['release_date']}")
        st.write(f"**🎞️🎥🍿 Genres:** {movie['genres']}")
        st.write(f"**👥 Cast:** {movie['cast']}")
        st.write(f"**🧑‍💼 Director:** {movie['director']}")
        st.write(f"**🎬 Overview:** {movie['overview']}")
        
        # Recommended movies
        st.subheader("Recommended Movies")
        rec_df = get_recommendations(movie_id, top_k=6)

        if rec_df.empty:
            st.write("No recommendations available.")
        else:
            rec_cols = st.columns(3)
            for i, (_, rec) in enumerate(rec_df.iterrows()):
                with rec_cols[i % 3]:
                    try:
                        if pd.notna(rec["poster_path"]) and rec["poster_path"] != "":
                            st.image(POSTER_BASE_URL + rec["poster_path"], use_container_width=True)
                        else:
                            st.image("default_poster.jpg", use_container_width=True)
                    except Exception as e:
                        st.write("🚫 Image Load Failed")
                    st.write(f"**{rec['title']}** ({rec['release_date']})")
                    if st.button(f"See Details: {rec['title']}", key=f"rec_{rec['id']}_{i}"):
                        st.session_state["selected_movie"] = rec["id"]
                        st.rerun()

# --------------------------
# 5. UI: Search Results Page
# --------------------------
 
if "search_results" in st.session_state:
    st.subheader("Search Results")
    results = st.session_state["search_results"]
    
    if results.empty:
        st.write("No search results found.")
    else:
        result_cols = st.columns(3)
        for i, (_, row) in enumerate(results.iterrows()):
            with result_cols[i % 3]:
                try:
                    if pd.notna(row["poster_path"]) and row["poster_path"] != "":
                        st.image(POSTER_BASE_URL + row["poster_path"], use_container_width=True)
                    else:
                        st.image("default_poster.jpg", use_container_width=True)
                except Exception as e:
                    st.write("🚫 Image Load Failed")
                st.write(f"**{row['title']}** ({row['release_date']})")
                if st.button(f"See Details: {row['title']}", key=f"search_{row['id']}"):  # Line 208
                    st.session_state["selected_movie"] = row["id"]

                    st.rerun()
