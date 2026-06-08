try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - provide a lightweight fallback for environments without streamlit
    # Minimal stub of streamlit API used in this app so linters/IDEs don't flag missing import
    from types import SimpleNamespace

    def _noop(*args, **kwargs):
        return None

    def _identity_decorator(func):
        return func

    class _Sidebar:
        def header(self, *a, **k):
            return None

        def selectbox(self, *a, **k):
            return a[1][0] if len(a) > 1 and isinstance(a[1], (list, tuple)) else (a[1] if len(a) > 1 else None)

    st = SimpleNamespace(
        set_page_config=_noop,
        title=_noop,
        write=_noop,
        cache_data=lambda *a, **k: _identity_decorator,
        cache_resource=lambda *a, **k: _identity_decorator,
        sidebar=_Sidebar(),
        selectbox=lambda *a, **k: (a[1][0] if len(a) > 1 and isinstance(a[1], (list, tuple)) else None),
        button=lambda *a, **k: False,
        subheader=_noop,
        dataframe=_noop,
        success=_noop,
        error=_noop
    )
import pandas as pd  # type: ignore
from pathlib import Path
try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover - provide clear error if sklearn is missing
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "scikit-learn is required for this app. Install it with: pip install scikit-learn"
            )

    MinMaxScaler = _MissingDependency
    NearestNeighbors = _MissingDependency

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Recommendation System")
st.write("Get song recommendations based on audio features.")

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

    # Inform the user in the Streamlit UI and allow file upload as a fallback
    try:
        st.error(
            "dataset.csv not found. Searched: " + ", ".join(str(p) for p in candidates)
        )
    except Exception:
        pass

    try:
        uploaded = st.file_uploader("Upload dataset.csv", type=["csv"])  # type: ignore
    except Exception:
        uploaded = None

    if uploaded is not None:
        return pd.read_csv(uploaded)

    raise FileNotFoundError(
        "dataset.csv not found. Searched: " + ", ".join(str(p) for p in candidates)
    )

df = load_data()

# -----------------------------
# Feature Selection
# -----------------------------
features = [
    'danceability',
    'energy',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'loudness',
    'popularity'
]

df = df.dropna(subset=features)
df = df.reset_index(drop=True)

# =====================================
# MOOD COLUMN
# =====================================

def get_mood(row):

    if row["energy"] > 0.7 and row["valence"] > 0.6:
        return "Happy"

    elif row["energy"] > 0.7:
        return "Workout"

    elif row["valence"] < 0.4:
        return "Sad"

    else:
        return "Chill"

df["mood"] = df.apply(get_mood, axis=1)

# -----------------------------
# Train KNN Model
# -----------------------------
@st.cache_resource
def train_model(data):

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])

    knn = NearestNeighbors(
        n_neighbors=11,
        metric='cosine'
    )

    knn.fit(X)

    return knn, X

knn, X = train_model(df)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(song_name):

    matches = df[
        df['track_name'].str.lower()
        == song_name.lower()
    ]

    if len(matches) == 0:
        return None

    idx = matches.index[0]

    distances, indices = knn.kneighbors(
        X[idx].reshape(1, -1)
    )

    recommendations = []

    for i in indices[0][1:]:

        recommendations.append({
            "Song": df.iloc[i]['track_name'],
            "Artist": df.iloc[i]['artists'],
            "Genre": df.iloc[i]['track_genre'],
            "Popularity": df.iloc[i]['popularity']
        })

    return pd.DataFrame(recommendations)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

genre_list = sorted(
    df['track_genre'].dropna().unique()
)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + list(genre_list)
)

# -----------------------------
# Song Selection
# -----------------------------
if selected_genre == "All":
    song_options = sorted(
        df['track_name']
        .dropna()
        .astype(str)
        .unique()
    )
else:
    song_options = sorted(
        df[df['track_genre'] == selected_genre]
        ['track_name']
        .dropna()
        .astype(str)
        .unique()
    )

selected_song = st.selectbox(
    "Choose a Song",
    song_options
)

# -----------------------------
# Recommendation Button
# -----------------------------
if st.button("🎧 Recommend Songs"):

    result = recommend(selected_song)

    if result is not None:

        st.success(
            f"Recommendations similar to '{selected_song}'"
        )

        st.dataframe(
            result,
            use_container_width=True
        )

    else:
        st.error("Song not found.")

# -----------------------------
# Popular Songs Section
# -----------------------------
st.subheader("🔥 Top Trending Songs")

top_songs = (
    df.sort_values(
        by="popularity",
        ascending=False
    )
    [['track_name',
      'artists',
      'track_genre',
      'popularity']]
    .head(10)
)

st.dataframe(
    top_songs,
    use_container_width=True
)

def recommend_by_mood(mood):

    return (
        df[df["mood"] == mood]
        .sort_values(
            "popularity",
            ascending=False
        )
        [["track_name","artists","track_genre"]]
        .head(10)
    )

st.sidebar.header("Filters")

feature = st.sidebar.radio(
    "Recommendation Type",
    [
        "Song Based",
        "Mood Based",
        "Trending Songs"
    ]
)

if feature == "Mood Based":

    mood = st.selectbox(
        "Select Mood",
        ["Happy","Sad","Workout","Chill"]
    )

    if st.button("Recommend"):
        st.dataframe(
            recommend_by_mood(mood)
        )

if feature == "Trending Songs":

    trending = (
        df.sort_values(
            "popularity",
            ascending=False
        )
        [["track_name",
            "artists",
            "track_genre",
            "popularity"]]
        .head(20)
    )

    st.dataframe(trending)

if selected_genre == "All":
    song_options = sorted(
        df['track_name']
        .dropna()
        .astype(str)
        .unique()
    )
else:
    song_options = sorted(
        df[df['track_genre'] == selected_genre]
        ['track_name']
        .dropna()
        .astype(str)
        .unique()
    )

st.markdown(
"""
# 🎵 Spotify Music Recommender

Discover songs based on:
- Similarity
- Mood
- Genre
- Popularity

"""
)