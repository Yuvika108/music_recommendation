import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Music Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background():
    base = Path(__file__).resolve().parent
    bg_files = [base / "background.jpg", base / "background.png", base / "background.jpeg"]
    for bg_file in bg_files:
        if bg_file.exists():
            bin_str = get_base64_of_bin_file(bg_file)
            ext = bg_file.suffix.replace('.', '')
            page_bg_img = f'''
            <style>
            [data-testid="stAppViewContainer"], .stApp {{
                background-image: url("data:image/{ext};base64,{bin_str}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-attachment: fixed !important;
            }}
            /* Add a semi-transparent dark overlay to ensure text remains readable */
            [data-testid="stHeader"] {{
                background-color: transparent !important;
            }}
            .main {{
                background-color: rgba(9, 9, 11, 0.7) !important; 
            }}
            /* Disable the previous radial gradients since we have an image now */
            .stApp::before, .stApp::after {{
                display: none !important;
            }}
            </style>
            '''
            st.markdown(page_bg_img, unsafe_allow_html=True)
            return

set_background()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        background-color: #09090b;
        color: #e4e4e7;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Subtle ambient glow */
    .stApp::before {
        content: '';
        position: fixed;
        top: -20%;
        left: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.05) 0%, rgba(0,0,0,0) 60%);
        z-index: -1;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -20%;
        right: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.04) 0%, rgba(0,0,0,0) 60%);
        z-index: -1;
    }
    
    /* Card Styles */
    .song-card {
        background: rgba(24, 24, 27, 0.5);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        transition: all 0.2s ease;
        display: flex;
        flex-direction: column;
    }
    
    .song-card:hover {
        background: rgba(39, 39, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .song-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #fafafa;
        line-height: 1.4;
        letter-spacing: -0.01em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-artist {
        font-size: 0.9rem;
        color: #a1a1aa;
        font-weight: 400;
        margin-bottom: 16px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-details {
        font-size: 0.75rem;
        color: #71717a;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #fafafa !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
    }
    p {
        color: #a1a1aa;
        font-size: 0.95rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #09090b !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: #71717a !important;
        font-weight: 500 !important;
        padding-bottom: 12px !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e4e4e7 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #fafafa !important;
        border-bottom: 2px solid #fafafa !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #fafafa !important;
        color: #09090b !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #e4e4e7 !important;
        transform: scale(0.99) !important;
    }
    
    /* Inputs */
    .stSelectbox > div > div {
        background-color: #18181b !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        color: #fafafa !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent
    candidates = [base / "dataset.csv", Path.cwd() / "dataset.csv"]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None

df = load_data()

if df is None:
    st.error("dataset.csv not found.")
    st.stop()

# -----------------------------
# Feature Selection & Preprocessing
# -----------------------------
audio_features = [
    'danceability', 'energy', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'loudness'
]
features_for_model = audio_features + ['popularity']

df = df.dropna(subset=features_for_model)
df = df.reset_index(drop=True)

def get_mood(row):
    if row["energy"] > 0.7 and row["valence"] > 0.6:
        return "Joyful"
    elif row["energy"] > 0.7:
        return "Energetic"
    elif row["valence"] < 0.4:
        return "Melancholic"
    else:
        return "Tranquil"

if "mood" not in df.columns:
    df["mood"] = df.apply(get_mood, axis=1)

# -----------------------------
# Train KNN Model
# -----------------------------
@st.cache_resource
def train_model(data):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features_for_model])
    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(X)
    return knn, X, scaler

knn, X, scaler = train_model(df)

# -----------------------------
# Helper Functions
# -----------------------------
def get_song_card_html(title, artist, genre, popularity):
    return f"""
    <div class="song-card">
        <div class="song-title">{title}</div>
        <div class="song-artist">{artist}</div>
        <div class="song-details">
            <span>{genre}</span>
            <span>Pop: {popularity}</span>
        </div>
    </div>
    """

def create_radar_chart(song_data1, song_name1, song_data2=None, song_name2=None):
    fig = go.Figure()

    plot_features = ['danceability', 'energy', 'acousticness', 'valence', 'liveness']
    
    val1 = [song_data1[f] for f in plot_features]
    val1 += [val1[0]]
    
    categories = [f.capitalize() for f in plot_features] + [plot_features[0].capitalize()]
    
    fig.add_trace(go.Scatterpolar(
        r=val1,
        theta=categories,
        fill='toself',
        name=song_name1,
        line_color='#818cf8', # Soft indigo
        fillcolor='rgba(129, 140, 248, 0.2)'
    ))

    if song_data2 is not None:
        val2 = [song_data2[f] for f in plot_features]
        val2 += [val2[0]]
        fig.add_trace(go.Scatterpolar(
            r=val2,
            theta=categories,
            fill='toself',
            name=song_name2,
            line_color='#c084fc', # Soft purple
            fillcolor='rgba(192, 132, 252, 0.2)'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color='rgba(255,255,255,0.1)', tickfont=dict(size=9)),
            angularaxis=dict(color='#a1a1aa', tickfont=dict(size=10))
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#fafafa', size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif'),
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

# -----------------------------
# Main UI
# -----------------------------
st.title("Music Explorer")
st.markdown("Discover your next favorite track.")
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Filters")
genre_list = sorted(df['track_genre'].dropna().unique())
selected_genre = st.sidebar.selectbox("Genre", ["All"] + list(genre_list))

if selected_genre == "All":
    filtered_df = df
else:
    filtered_df = df[df['track_genre'] == selected_genre]

# Tabs
tab1, tab2, tab3 = st.tabs(["Similar Tracks", "Mood", "Trending"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    song_options = sorted(filtered_df['track_name'].dropna().astype(str).unique())
    selected_song = st.selectbox("Search for a track", song_options)

    if st.button("Find Matches", use_container_width=True):
        matches = filtered_df[filtered_df['track_name'].str.lower() == selected_song.lower()]
        
        if len(matches) > 0:
            original_idx = df[df['track_name'].str.lower() == selected_song.lower()].index[0]
            distances, indices = knn.kneighbors(X[original_idx].reshape(1, -1))
            
            st.markdown("<br>### Recommendations", unsafe_allow_html=True)
            
            rec_indices = indices[0][1:7]
            cols = st.columns(3)
            
            for i, rec_idx in enumerate(rec_indices):
                rec_row = df.iloc[rec_idx]
                with cols[i % 3]:
                    st.markdown(
                        get_song_card_html(
                            rec_row['track_name'],
                            rec_row['artists'],
                            rec_row['track_genre'],
                            rec_row['popularity']
                        ),
                        unsafe_allow_html=True
                    )
            
            st.markdown("<br>---<br>", unsafe_allow_html=True)
            st.markdown("### Audio Profile")
            st.markdown("<p style='font-size:0.9rem; color:#a1a1aa;'>Comparing the acoustic features of your selected track with the top recommendation.</p>", unsafe_allow_html=True)
            
            col_chart1, col_chart2 = st.columns([1, 2])
            top_rec_row = df.iloc[rec_indices[0]]
            
            with col_chart1:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"**Selected:**<br><span style='color:#a1a1aa'>{selected_song}</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Top Match:**<br><span style='color:#a1a1aa'>{top_rec_row['track_name']}</span>", unsafe_allow_html=True)
                
            with col_chart2:
                fig = create_radar_chart(
                    df.iloc[original_idx], selected_song,
                    top_rec_row, top_rec_row['track_name']
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Track not found.")

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    mood = st.selectbox("How are you feeling?", ["Joyful", "Melancholic", "Energetic", "Tranquil"])
    
    if st.button("Show Tracks", use_container_width=True):
        mood_df = filtered_df[filtered_df["mood"] == mood].sort_values("popularity", ascending=False).head(6)
        
        if len(mood_df) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (_, row) in enumerate(mood_df.iterrows()):
                with cols[i % 3]:
                    st.markdown(
                        get_song_card_html(
                            row['track_name'],
                            row['artists'],
                            row['track_genre'],
                            row['popularity']
                        ),
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No tracks found.")

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    trending = filtered_df.sort_values("popularity", ascending=False).head(9)
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(trending.iterrows()):
        with cols[i % 3]:
             st.markdown(
                get_song_card_html(
                    row['track_name'],
                    row['artists'],
                    row['track_genre'],
                    row['popularity']
                ),
                unsafe_allow_html=True
            )