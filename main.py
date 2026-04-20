from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DATA LOAD
# =========================
df = pd.read_csv("model/data.csv")

# Duplicate temizleme
df = df.drop_duplicates(subset=["track_name", "artists"])
df = df.reset_index(drop=True)

# =========================
# FEATURE LIST
# =========================
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

# =========================
# SCALER LOAD
# =========================
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

scaled_features = scaler.transform(df[features])

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(song_index, top_n=5):
    song_vector = scaled_features[song_index].reshape(1, -1)

    audio_sim = cosine_similarity(song_vector, scaled_features)[0]
    genre_sim = (df["track_genre"] == df.iloc[song_index]["track_genre"]).astype(int)

    final_score = 0.9 * audio_sim + 0.1 * genre_sim

    top_indices = np.argsort(final_score)[::-1][1:top_n+1]

    return df.iloc[top_indices][["track_name", "artists"]].to_dict(orient="records")


# =========================
# ROOT (TEST)
# =========================
@app.get("/")
def home():
    return {"message": "Music Recommendation API is running 🎧"}


# =========================
# SEARCH + MATCH
# =========================
@app.get("/recommend")
def recommend_api(song: str):
    matches = df[df["track_name"].str.contains(song, case=False, na=False)]

    # duplicate temizleme
    matches = matches.drop_duplicates(subset=["track_name", "artists"])

    if len(matches) == 0:
        return {"error": "Song not found"}

    if len(matches) > 1:
        # 🔥 SADECE GEREKLİ KOLONLAR
        matches = matches[["track_name", "artists"]].head(10)

        # index ekle
        matches["index"] = matches.index

        return {
            "matches": matches.to_dict(orient="records")
        }

    song_index = matches.index[0]

    return {
        "recommendations": recommend(song_index)
    }


# =========================
# DIRECT INDEX RECOMMEND
# =========================
@app.get("/recommend-by-index")
def recommend_by_index(index: int):
    if index >= len(df):
        return {"error": "Invalid index"}

    return {
        "recommendations": recommend(index)
    }