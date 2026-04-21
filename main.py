from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import base64
import os
import urllib.parse
import time


app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spotify_token = None
spotify_token_expires = 0
cover_cache = {}

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

def get_spotify_token():
    global spotify_token, spotify_token_expires

    if spotify_token and time.time() < spotify_token_expires:
        return spotify_token

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    url = "https://accounts.spotify.com/api/token"

    headers = {
        "Authorization": f"Basic {auth_base64}"
    }

    data = {
        "grant_type": "client_credentials"
    }

    result = requests.post(url, headers=headers, data=data, timeout=5)
    json_result = result.json()

    spotify_token = json_result["access_token"]
    spotify_token_expires = time.time() + json_result["expires_in"]

    return spotify_token

def get_album_cover(track_name, artist):
    key = f"{track_name.lower()}_{artist.lower()}"

    if key in cover_cache:
        return cover_cache[key]

    token = get_spotify_token()

    query = urllib.parse.quote(f"{track_name} {artist}")
    url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers, timeout=5)
    data = response.json()

    try:
        image = data["tracks"]["items"][0]["album"]["images"][0]["url"]
        cover_cache[key] = image
        return image
    except:
        return "https://via.placeholder.com/300"




# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(song_index, top_n=5):
    song_vector = scaled_features[song_index].reshape(1, -1)

    audio_sim = cosine_similarity(song_vector, scaled_features)[0]
    genre_sim = (df["track_genre"] == df.iloc[song_index]["track_genre"]).astype(int)

    final_score = 0.9 * audio_sim + 0.1 * genre_sim

    top_indices = np.argsort(final_score)[::-1][1:top_n+1]

    results = df.iloc[top_indices][["track_name", "artists"]].to_dict(orient="records")

    for r in results:
        r["image"] = get_album_cover(r["track_name"], r["artists"])

    return results


# =========================
# ROOT (TEST)
# =========================
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


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
        matches = matches[["track_name", "artists"]].head(10)
        matches["index"] = matches.index

        matches = matches.to_dict(orient="records")

        for m in matches:
            m["image"] = get_album_cover(m["track_name"], m["artists"])

        return {"matches": matches}

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