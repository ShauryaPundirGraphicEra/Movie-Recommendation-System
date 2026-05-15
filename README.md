
# 🎬 Cine-Vector: Semantic Movie Recommendation Engine

![Python](https://img.shields.io/badge/Language-Python_3.12-blue)
![SentenceTransformers](https://img.shields.io/badge/ML-Sentence_Transformers-orange)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-brightgreen)
![TMDB](https://img.shields.io/badge/API-TMDB_Integration-yellow)
![Pandas](https://img.shields.io/badge/Data-Pandas-lightgrey)

**Cine-Vector** is an AI-powered movie recommendation system that leverages Natural Language Processing (NLP) and Vector Search to deliver highly contextual movie suggestions. Moving beyond traditional collaborative filtering, this project utilizes **Sentence-Transformer embeddings** and **ChromaDB** to understand the deep semantic relationships between movie plots, genres, cast, and directors.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key AI / ML Features](#-key-aiml-features)
3. [System Architecture & ETL](#-system-architecture--etl)
4. [Tech Stack](#-tech-stack)
5. [Installation & Local Setup](#-installation--local-setup)
6. [How the Recommendation Engine Works](#-how-the-recommendation-engine-works)
7. [Future Scope](#-future-scope)

---

## 🎯 Project Overview
Traditional recommendation engines often suffer from the "Cold Start" problem and rely heavily on sparse user-item interaction matrices. 

**Cine-Vector** solves this by taking a purely **content-based, semantic approach**. By extracting thousands of records from the TMDB API and transforming rich textual metadata (plots, cast, directors, genres) into high-dimensional vectors, the system can instantly identify and recommend movies that share thematic, narrative, or stylistic DNA—even if the user has no prior watch history.

---

## 🧠 Key AI/ML Features
* **Automated Data Engineering Pipeline:** Programmatically paginates through the TMDB API to fetch, clean, and consolidate records for over 10,000 movies, including dynamic resolution of genre IDs and nested cast/crew JSON arrays.
* **Dense Text Representations:** Constructs a unified `text_for_embedding` feature by intelligently concatenating "Movie Title", "Overview/Plot", "Cast", "Genres", and "Director" into a single, context-rich document.
* **Sentence-Transformer Embeddings:** Utilizes HuggingFace's `sentence_transformers` library to convert textual metadata into dense numerical vectors, capturing the nuanced semantic meaning of a film's narrative.
* **High-Performance Vector Search:** Integrates **ChromaDB** to store and index the generated embeddings, enabling lightning-fast similarity search (e.g., Cosine Similarity/L2 distance) to retrieve the Top-K most relevant movies instantly.

---

## 🏗 System Architecture & ETL

The application is structured into a logical, reproducible Data Science pipeline:

1. **Extraction (TMDB API):** The `Fetching_Api_For_Movie.ipynb` notebook acts as the ETL orchestrator. It pulls 500 pages of movie data, fetches subsequent credits (Cast/Director), and normalizes the data into a structured Pandas DataFrame.
2. **Transformation & Feature Engineering:** Drops null values and orchestrates the creation of the aggregated context string (`text_for_embedding`).
3. **Loading & Indexing:** The `Movie_Recommendation_System (1).ipynb` loads the clean `full_moviess.csv`, generates the embeddings via PyTorch-backed SentenceTransformers, and loads them into the local ChromaDB vector store.
4. **Application Interface (`app.py`):** The main execution script that loads the trained model/database to serve user queries interactively.

---

## 💻 Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Language & Data Processing** | Python 3.12, Pandas, NumPy, Requests |
| **Machine Learning / NLP** | PyTorch, Sentence-Transformers (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Data Source** | TMDB (The Movie Database) API V3 |
| **Environment** | Jupyter Notebooks / Google Colab |

---

## 🚀 Installation & Local Setup

### Prerequisites
* Python (3.9+)
* TMDB API Key (Create an account on TMDB to get your v3 Auth Key)

### Step 1: Clone the Repository
```bash
git clone [https://github.com/YourUsername/Cine-Vector-Recommendation.git](https://github.com/YourUsername/Cine-Vector-Recommendation.git)
cd Cine-Vector-Recommendation

```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

```

*(Ensure that packages like `pandas`, `requests`, `sentence_transformers`, and `chromadb` are installed.)*

### Step 3: Run the Data Pipeline

1. Open `Fetching_Api_For_Movie.ipynb`.
2. Insert your TMDB API Key in the `api_key` variable.
3. Run all cells to fetch the latest movie data and generate the `full_moviess.csv` file.

### Step 4: Generate Vector Embeddings

1. Open `Movie_Recommendation_System (1).ipynb`.
2. Run the cells to process the `full_moviess.csv` file, build the `text_for_embedding` column, and encode the data using `sentence_transformers` into `chromadb`.

### Step 5: Launch the App

```bash
python app.py

```

---

## 🧩 How the Recommendation Engine Works

### 1. The Context Aggregation Phase

Instead of treating "Genres" and "Plot" as isolated categorical variables, the system creates a cohesive narrative block for the AI to read.
Example:

> *"Movie Title: Laila. Movie Overview/Plot: Sonu Model, a renowned beautician... Cast: ['Vishwak Sen', 'Akanksha Sharma'] Genres: ['Comedy', 'Romance']. Director: ['Ram Narayan']"*

### 2. The Embedding Phase

The combined text is passed through a pre-trained transformer model (via `sentence_transformers`). The attention mechanisms inside the transformer analyze the relationships between the words (e.g., associating "space" and "alien" with the "Sci-Fi" genre), outputting a dense vector.

### 3. The Retrieval Phase

When a user selects a movie they like, the system queries **ChromaDB**. The database calculates the mathematical distance between the chosen movie's vector and all other 10,000+ movie vectors in the database. The closest vectors (nearest neighbors) are returned as the Top-K recommendations.

---

## 🔮 Future Scope

* **Hybrid Filtering:** Combine the current Semantic Vector Search with Collaborative Filtering (user-rating matrices) to weigh recommendations based on both plot similarity and community popularity.
* **Real-time API Integration:** Move away from static CSV generation and fetch/embed new TMDB releases dynamically via a daily cron job.
* **Frontend Web Application:** Wrap the `app.py` logic in a React.js or Next.js frontend with TMDB poster rendering for a Netflix-style UI.

---

```

```


DEMO Link:- https://youtu.be/5dlmTCzqOJM?si=5vc6mIyif6P7wHd8

Sample :-

<img width="2866" height="1686" alt="image" src="https://github.com/user-attachments/assets/5670dabc-0972-43c1-b8fb-00895fa8ce0a" />



