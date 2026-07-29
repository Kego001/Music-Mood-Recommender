# 🎵 Music Mood Recommender

Find songs similar to what you love — with mood control.

## What it does
- Search any song by name and artist
- Get 10 similar song recommendations
- Adjust mood using Energy and Happiness sliders
- 30 second previews via iTunes
- Direct Spotify links for every recommendation

## How it works
- 114,000+ songs dataset with audio features (energy, valence, danceability etc.)
- Cosine similarity to find mathematically similar songs
- Mood adjustment by modifying energy/valence in feature vector before similarity search
- Spotify API for song search and album covers
- iTunes API for audio previews

## Tech used
Python, Streamlit, Scikit-learn, Spotipy, Pandas, NumPy

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Dataset
Spotify Tracks Dataset — 114k songs from Kaggle