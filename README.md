# 🎧 MelodyMatch – Music Recommendation System

MelodyMatch is a full-stack music recommendation system that suggests similar songs based on audio features using machine learning.

The system allows users to search for a song, select the correct match, and instantly receive recommendations with a modern and interactive UI.

---

## 🚀 Live Demo

👉 [Live Application](https://songrecommendation-production.up.railway.app/)

---

## 📌 Features

* 🔍 Real-time song search (autocomplete with debounce)
* 🎯 Smart song matching system
* 🎧 Music recommendations using similarity algorithms
* ⚡ FastAPI backend with REST API
* 🌐 Fully deployed web application (Railway)
* 💻 Modern responsive frontend UI
* 🔗 Single-link access (frontend + backend integrated)

---

## 🧠 How It Works

The recommendation system is based on **content-based filtering** using audio features:

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo

### Algorithm:

* Features are scaled using `StandardScaler`
* Cosine similarity is computed between songs
* Genre similarity is added as a small boost

```text
Final Score = 0.9 * Audio Similarity + 0.1 * Genre Match
```

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* Scikit-learn
* Pandas / NumPy

### Frontend

* HTML / CSS / JavaScript
* Vanilla JS (no framework)

### Deployment

* Railway (backend + frontend hosting)

---

## 📂 Project Structure

```
music-recommender/
│
├── main.py
├── requirements.txt
├── model/
│   ├── data.csv
│   ├── scaler.pkl
│
├── frontend/
│   └── index.html
```

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/bursal0/SongRecommendation.git
cd SongRecommendation

pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

### 🔍 Search Songs

```
GET /recommend?song=your_song
```

### 🎧 Get Recommendations by Index

```
GET /recommend-by-index?index=0
```

---

## 💡 Future Improvements

* 🎵 Spotify API integration (album covers & metadata)
* 👤 User-based personalization
* 🤖 Deep learning / embeddings
* 📊 Better ranking models

---

## 📜 License

This project is open-source and available for educational purposes.

---

## 👤 Author

Developed by Ahmet Yusuf Bursal
Computer Engineering Student | Machine Learning Enthusiast

---
