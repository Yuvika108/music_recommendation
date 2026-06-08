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
    page_title="Music Recommendation System",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    /* Import vintage fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Merriweather:wght@300;400;700&display=swap');

    /* Vintage parchment background */
    .stApp {
        background-color: #F4EAD5;
        color: #3E2723;
        font-family: 'Merriweather', 'Times New Roman', serif;
    }
    
    /* Card Styles: Elegant vintage frames */
    .song-card {
        background-color: #FAEDE1;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
        border: 4px solid #8D6E63;
        border-style: double;
        box-shadow: 3px 3px 6px rgba(62,39,35,0.15);
        transition: transform 0.2s ease-in-out, background-color 0.2s ease-in-out;
    }
    .song-card:hover {
        transform: translateY(-3px);
        background-color: #FDF3E9;
        box-shadow: 5px 5px 10px rgba(62,39,35,0.2);
    }
    .song-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4E342E;
        margin-bottom: 8px;
        line-height: 1.2;
        font-family: 'Playfair Display', serif;
    }
    .song-artist {
        font-size: 1.1rem;
        color: #5D4037;
        margin-bottom: 8px;
        font-style: italic;
    }
    .song-details {
        font-size: 0.9rem;
        color: #8D6E63;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        font-family: 'Playfair Display', serif;
        border-top: 1px dashed #D7CCC8;
        padding-top: 8px;
        margin-top: 8px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #3E2723 !important;
        font-family: 'Playfair Display', serif;
        border-bottom: 1px solid #D7CCC8;
        padding-bottom: 8px;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #EAE0C8;
        border-right: 3px double #8D6E63;
    }

    /* Tabs styling to match vintage look */
    button[data-baseweb="tab"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem;
        color: #5D4037 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3E2723 !important;
        border-bottom-color: #3E2723 !important;
        font-weight: bold;
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
    st.error("dataset.csv not found. Please ensure the dataset is in the same directory.")
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

# Define Moods
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
        <div class="song-title">❦ {title}</div>
        <div class="song-artist">By {artist}</div>
        <div class="song-details">
            <span>✧ Genre: {genre}</span>
            <span>⚑ Popularity: {popularity}</span>
        </div>
    </div>
    """

def create_radar_chart(song_data1, song_name1, song_data2=None, song_name2=None):
    fig = go.Figure()

    # Normalize data for radar chart visualization
    plot_features = ['danceability', 'energy', 'acousticness', 'valence', 'liveness']
    
    val1 = [song_data1[f] for f in plot_features]
    val1 += [val1[0]] # Close the polygon
    
    categories = plot_features + [plot_features[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=val1,
        theta=categories,
        fill='toself',
        name=song_name1,
        line_color='#8D6E63' # Vintage Brown
    ))

    if song_data2 is not None:
        val2 = [song_data2[f] for f in plot_features]
        val2 += [val2[0]]
        fig.add_trace(go.Scatterpolar(
            r=val2,
            theta=categories,
            fill='toself',
            name=song_name2,
            line_color='#C2185B' # Vintage Burgundy
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(color='#3E2723')
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#3E2723', family='Playfair Display, serif')
    )
    return fig

# -----------------------------
# Main UI
# -----------------------------
st.title("📜 Melody Archives")
st.markdown("Unearth your next beloved composition through our distinguished catalog of musical works.")

# Sidebar Filters
st.sidebar.title("⚜️ Compendium Filters")
genre_list = sorted(df['track_genre'].dropna().unique())
selected_genre = st.sidebar.selectbox("Select a Genre", ["All"] + list(genre_list))

if selected_genre == "All":
    filtered_df = df
else:
    filtered_df = df[df['track_genre'] == selected_genre]

# Tabs
tab1, tab2, tab3 = st.tabs(["☙ Kinship Inquiry", "❦ Temperament", "♛ Acclaimed Works"])

# TAB 1: Song Similarity
with tab1:
    st.header("Seek Kindred Compositions")
    song_options = sorted(filtered_df['track_name'].dropna().astype(str).unique())
    selected_song = st.selectbox("Inquire regarding a composition you favor", song_options)

    if st.button("Unveil Similar Works", use_container_width=True):
        matches = filtered_df[filtered_df['track_name'].str.lower() == selected_song.lower()]
        
        if len(matches) > 0:
            # Must get index in the ORIGINAL dataframe because X is fitted on original df
            original_idx = df[df['track_name'].str.lower() == selected_song.lower()].index[0]
            
            distances, indices = knn.kneighbors(X[original_idx].reshape(1, -1))
            
            st.markdown("### Most Comparable Discoveries")
            
            rec_indices = indices[0][1:7] # Top 6 recommendations
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
            
            st.markdown("---")
            st.markdown("### ✒️ Acoustic Analysis")
            st.markdown("An examination of the auditory characteristics of your inquiry against the foremost recommendation.")
            
            col_chart1, col_chart2 = st.columns([1, 2])
            
            top_rec_row = df.iloc[rec_indices[0]]
            
            with col_chart1:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.info(f"**Inquiry:** {selected_song}")
                st.success(f"**Foremost Discovery:** {top_rec_row['track_name']}")
                
            with col_chart2:
                fig = create_radar_chart(
                    df.iloc[original_idx], selected_song,
                    top_rec_row, top_rec_row['track_name']
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Alas, the composition could not be found within the archives.")

# TAB 2: Mood Matcher
with tab2:
    st.header("Match Your Temperament")
    mood = st.selectbox("What is your current disposition?", ["Joyful", "Melancholic", "Energetic", "Tranquil"])
    
    if st.button("Discover Suitable Works", use_container_width=True):
        mood_df = filtered_df[filtered_df["mood"] == mood].sort_values("popularity", ascending=False).head(6)
        
        if len(mood_df) > 0:
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
            st.warning("No compositions were found to match this temperament and genre.")

# TAB 3: Trending
with tab3:
    st.header("♛ Foremost Acclaimed Works")
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