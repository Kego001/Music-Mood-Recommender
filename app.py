import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fuzzywuzzy import fuzz
import json
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Music Mood Recommender", page_icon="🎵")

st.title("🎵 Music Mood Recommender")
st.write("Find songs similar to what you love — with mood control")

@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    df = df.dropna()
    
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo']
    
    df = df[['track_id', 'artists', 'track_name', 'track_genre'] + feature_cols]
    
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols

df, feature_cols = load_data()

# spotify setup
@st.cache_resource
def setup_spotify():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
    ))
    return sp
sp = setup_spotify()

# user inputs
st.subheader("Search a Song")
song_name = st.text_input("Song name", placeholder="e.g. Believer")
artist_name = st.text_input("Artist name", placeholder="e.g. Imagine Dragons")
print(song_name)

st.subheader("Mood Adjustment")

col1, col2 = st.columns(2)
with col1:
    energy_adjust = st.slider("Energy", -1.0, 1.0, 0.0, 0.1)   
with col2:
    valence_adjust = st.slider("Happiness", -1.0, 1.0, 0.0, 0.1)

recommend_btn = st.button("Find Songs 🎵")

def get_song_features(song_name, artist_name, df, feature_cols):
    if artist_name: results = sp.search(q=f"track:{song_name} artist:{artist_name}", limit=1)
    else: results = sp.search(q=f"track:{song_name}", limit=1)
    if not results['tracks']['items']:
        return None, None
    
    track=results['tracks']['items'][0]
    correct_name=track['name']
    correct_artist=track['artists'][0]['name']
    
    print(f"Found: {correct_name} by {correct_artist}")
    
    # now look up in dataset
    mask=((df['track_name'].str.lower() == correct_name.lower()) & (df["artists"].str.lower().str.contains(correct_artist.lower())))
    results_df = df[mask]
    
    if len(results_df) == 0:
        track_id = results['tracks']['items'][0]['id']
        return f"NOTFOUND:{correct_name}:{correct_artist}:{track_id}",None
    
    song = results_df.iloc[0]
    return song, song.name

def recommend_songs(song_name,artist_name,df,feature_cols,n=10,energy_adjust=0, valence_adjust=0):
    song,song_idx=get_song_features(song_name, artist_name, df, feature_cols)
    if song is None:
        return "Song not available"
    if isinstance(song, str) and song.startswith("NOTFOUND:"):
        return song

    genre=song['track_genre']
    genre_df=df[df['track_genre']==genre].copy()

    song_vector=song[feature_cols].values.reshape(1, -1).copy()

    energy_idx = feature_cols.index('energy')
    valence_idx = feature_cols.index('valence')
    
    song_vector[0][energy_idx] = np.clip(song_vector[0][energy_idx] + energy_adjust, 0, 1)
    song_vector[0][valence_idx] = np.clip(song_vector[0][valence_idx] + valence_adjust, 0, 1)

    genre_matrix=genre_df[feature_cols].values

    similarities=cosine_similarity(song_vector, genre_matrix)[0]

    similar_indices = similarities.argsort()[::-1]
    results = genre_df.iloc[similar_indices][['track_name', 'artists', 'track_genre']]
    #results = results[results.index != song_idx]
    results=results.drop_duplicates(subset=['track_name', 'artists'])[1:]

    results=results.head(n)
    return results

def get_spotify_info(track_name, artist_name):
    results = sp.search(q=f"track:{track_name} artist:{artist_name}",limit=1)
    if not results["tracks"]["items"]:
        return None

    track = results["tracks"]["items"][0]
    return {"spotify_link": track["external_urls"]["spotify"],
        "album_cover": track["album"]["images"][0]["url"] if track["album"]["images"] else None}

def get_itunes_preview(track_name, artist_name):
    import requests
    query = f"{track_name} {artist_name}".replace(" ", "+")
    url = f"https://itunes.apple.com/search?term={query}&media=music&limit=1"
    response = requests.get(url).json()
    
    if response['resultCount'] > 0:
        return response['results'][0].get('previewUrl')
    return None

@st.cache_data
def get_spotify_info_cached(track_name, artist_name):
    cache_file = 'spotify_cache.json'
    key = f"{track_name}_{artist_name}".lower()
    
    # load existing cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    
    # return if already cached
    if key in cache:
        return cache[key]
    
    # fetch and save
    info = get_spotify_info(track_name, artist_name)
    cache[key] = info
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    return info

if recommend_btn:
    if song_name: #and artist_name:
        with st.spinner("Finding songs..."):
            # get recommendations
            recommendations = recommend_songs(
                song_name, artist_name, df, feature_cols,
                energy_adjust=energy_adjust,
                valence_adjust=valence_adjust
            )
        
        if isinstance(recommendations, str):
            if recommendations.startswith("NOTFOUND:"):
                _, name, artist,track_id = recommendations.split(":")
                splink = f"https://open.spotify.com/track/{track_id}"
                st.warning(f"Coundn't get any recommendations for '{name}' by {artist} .")
                st.markdown(f"[Listen on Spotify]({splink})")
            else:
                st.error("Song not found anywhere.")
        else:
            st.subheader("Recommended Songs")
            for i,(_, row) in enumerate(recommendations.iterrows()):
                info = get_spotify_info_cached(row['track_name'], row['artists'])
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if info and info['album_cover']:
                            st.image(info['album_cover'], width=70)
                    with col2:
                        st.markdown(f"**{row['track_name']}** by {row['artists']}")
                        st.caption(f"Genre: {row['track_genre']}")
                        if info and info['spotify_link']:
                            st.markdown(f"[🎵 Play on Spotify]({info['spotify_link']})")
                        preview = get_itunes_preview(row['track_name'], row['artists'])
                        if preview:
                            st.audio(preview, format='audio/mp4')
                    st.divider()           
    else:
        st.warning("Please enter song name")
