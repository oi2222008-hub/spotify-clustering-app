import streamlit as st
import pandas as pd
import joblib
import base64
import streamlit.components.v1 as components
from urllib.parse import quote_plus

from sklearn.metrics.pairwise import euclidean_distances


def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def spotify_search_url(song, artist=""):
    query = quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{query}"
st.set_page_config(
    page_title="Spotify AI Song Cluster",
    page_icon="🎵",
    layout="wide"
)

st.markdown(
    """
    <style>
   html, body, [data-testid="stAppViewContainer"], .stApp {
    color: white !important;
    background:
        radial-gradient(circle at 20% 20%, rgba(0, 180, 255, 0.25), transparent 25%),
        radial-gradient(circle at 80% 30%, rgba(120, 0, 255, 0.20), transparent 30%),
        radial-gradient(circle at 50% 80%, rgba(255, 255, 255, 0.10), transparent 20%),
        linear-gradient(180deg, #02020a 0%, #050012 45%, #000000 100%) !important;
    background-size: 200% 200%;
    animation: rollingNightSky 22s ease-in-out infinite;
}

@keyframes rollingNightSky {
    0% {
        background-position: 0% 0%;
    }
    50% {
        background-position: 100% 80%;
    }
    100% {
        background-position: 0% 0%;
    }
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.55;
    background-image:
        radial-gradient(circle at 7% 12%, white 0.8px, transparent 1.8px),
        radial-gradient(circle at 18% 42%, rgba(255,255,255,0.9) 1.2px, transparent 2.2px),
        radial-gradient(circle at 29% 75%, rgba(173,216,230,0.9) 1px, transparent 2px),
        radial-gradient(circle at 36% 18%, white 1.5px, transparent 2.5px),
        radial-gradient(circle at 48% 61%, rgba(255,255,255,0.8) 0.9px, transparent 1.9px),
        radial-gradient(circle at 57% 31%, rgba(173,216,230,0.85) 1.4px, transparent 2.4px),
        radial-gradient(circle at 66% 84%, white 1px, transparent 2px),
        radial-gradient(circle at 76% 22%, rgba(255,255,255,0.95) 1.6px, transparent 2.6px),
        radial-gradient(circle at 87% 68%, rgba(173,216,230,0.9) 1.1px, transparent 2.1px),
        radial-gradient(circle at 94% 38%, white 0.8px, transparent 1.8px);
    background-size:
        260px 260px,
        340px 340px,
        420px 420px,
        300px 300px,
        460px 460px,
        380px 380px,
        520px 520px,
        330px 330px,
        440px 440px,
        280px 280px;
    animation: starsMove 24s linear infinite;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.35;
    background-image:
        radial-gradient(circle at 11% 83%, white 0.7px, transparent 1.7px),
        radial-gradient(circle at 25% 28%, rgba(173,216,230,0.9) 1.3px, transparent 2.3px),
        radial-gradient(circle at 41% 49%, white 1px, transparent 2px),
        radial-gradient(circle at 59% 12%, rgba(255,255,255,0.85) 1.4px, transparent 2.4px),
        radial-gradient(circle at 73% 77%, rgba(173,216,230,0.85) 0.9px, transparent 1.9px),
        radial-gradient(circle at 91% 54%, white 1.5px, transparent 2.5px);
    background-size:
        390px 390px,
        470px 470px,
        310px 310px,
        530px 530px,
        360px 360px,
        450px 450px;
    animation: starsMoveSide 32s linear infinite;
}

@keyframes starsMove {
    from {
        transform: translateY(-120px);
    }
    to {
        transform: translateY(260px);
    }
}

@keyframes starsMoveSide {
    from {
        transform: translate(-80px, -80px);
    }
    to {
        transform: translate(120px, 260px);
    }
}

.block-container {
    position: relative;
    z-index: 1;
}

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #050505 !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: white !important;
    }

    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        background: linear-gradient(90deg, #1DB954, #00FFFF, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cccccc !important;
        margin-bottom: 30px;
    }

    .glass-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 25px;
        padding: 25px;
        box-shadow: 0 0 35px rgba(29,185,84,0.3);
        margin-top: 20px;
    }

   .reaper-scene {
    position: relative;
    width: min(520px, 90vw);
    margin: 10px auto 30px auto;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    animation: floatReaper 4s ease-in-out infinite;
    z-index: 2;
    overflow: visible;
}

.reaper-img {
    width: 100%;
    display: block;
    position: relative;
    z-index: 3;
    background: transparent !important;
}

@keyframes floatReaper {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-22px);
    }
    100% {
        transform: translateY(0px);
    }
}

@keyframes floatReaper {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-22px);
    }
    100% {
        transform: translateY(0px);
    }
}

   .reaper-img {
    width: 100%;
    display: block;
    background: transparent !important;
}

   .pearl-overlay {
    position: absolute;
    top: 40%;
    left: 65%;
    transform: translate(-50%, -50%);
    width: 42%;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    pointer-events: none;
}

   .pearl-song {
    font-size: clamp(16px, 2vw, 28px);
    font-weight: 900;
    color: white !important;
    text-shadow:
        0 0 8px black,
        0 0 14px rgba(0, 220, 255, 0.9),
        0 0 22px rgba(255, 255, 255, 0.7);
    line-height: 1.1;
    max-width: 100%;
    overflow-wrap: break-word;
}
    .pearl-artist {
    font-size: clamp(11px, 1.2vw, 16px);
    margin-top: 8px;
    color: #e8f7ff !important;
    text-shadow: 0 0 8px black;
    max-width: 100%;
    overflow-wrap: break-word;
}

@keyframes floatReaper {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-22px);
    }
    100% {
        transform: translateY(0px);
    }
}
/* Number of recommendations slider - night blue thin line */

/* Main slider rail / line */
.stSlider [data-baseweb="slider"] div[role="presentation"] {
    background-color: #071a3d !important;
}

/* Active filled line */
.stSlider [data-baseweb="slider"] div[style*="background"] {
    background-color: #0b3d91 !important;
}

/* Extra force for the slider line */
.stSlider [data-baseweb="slider"] > div > div {
    background: #0b3d91 !important;
}

/* Slider handle / white ball */
.stSlider [role="slider"] {
    background: #ffffff !important;
    border: 3px solid #0b3d91 !important;
    box-shadow: 0 0 16px rgba(11, 61, 145, 0.95) !important;
    opacity: 1 !important;
}

/* Keep the ball white when selected */
.stSlider [role="slider"]:hover,
.stSlider [role="slider"]:active,
.stSlider [role="slider"]:focus,
.stSlider [role="slider"]:focus-visible {
    background: #ffffff !important;
    border: 3px solid #0b3d91 !important;
    box-shadow: 0 0 18px rgba(11, 61, 145, 1) !important;
    opacity: 1 !important;
    outline: none !important;
}

/* Remove red/orange browser accent */
.stSlider * {
    accent-color: #0b3d91 !important;
}
/* Make Choose a song area transparent */
.stSelectbox,
.stSelectbox > div,
.stSelectbox div[data-baseweb="select"],
.stSelectbox div[data-baseweb="select"] > div {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Make input/select field transparent */
div[data-baseweb="select"] {
    background: transparent !important;
}

div[data-baseweb="select"] > div {
    background: rgba(0, 0, 0, 0.15) !important;
    border: 1px solid rgba(173, 216, 230, 0.45) !important;
    box-shadow: 0 0 12px rgba(173, 216, 230, 0.25) !important;
}

/* Make section/card backgrounds transparent */
.glass-card {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Make dataframe/table container more transparent */
div[data-testid="stDataFrame"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Make headings transparent behind text */
h1, h2, h3, h4, h5, h6 {
    background: transparent !important;
}
    </style>
    """,
    unsafe_allow_html=True
)
components.html(
    """
    <div style="
        position: fixed;
        top: 18px;
        right: 24px;
        z-index: 999999;
    ">
        <button id="soundBtn" style="
            width: 58px;
            height: 58px;
            border-radius: 50%;
            border: 1px solid rgba(0, 220, 255, 0.8);
            background: rgba(0, 0, 0, 0.75);
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 0 18px rgba(0, 220, 255, 0.8);
        ">
            🔇
        </button>
    </div>

    <script>
    let audioStarted = false;
    let isMuted = true;
    let audioContext;
    let masterGain;
    let osc1;
    let osc2;
    let lfo;
    let lfoGain;

    const button = document.getElementById("soundBtn");

    function startOminousSound() {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        masterGain = audioContext.createGain();
        masterGain.gain.value = 0.0;
        masterGain.connect(audioContext.destination);

        osc1 = audioContext.createOscillator();
        osc1.type = "sine";
        osc1.frequency.value = 55;

        osc2 = audioContext.createOscillator();
        osc2.type = "triangle";
        osc2.frequency.value = 82;

        lfo = audioContext.createOscillator();
        lfo.type = "sine";
        lfo.frequency.value = 0.18;

        lfoGain = audioContext.createGain();
        lfoGain.gain.value = 18;

        lfo.connect(lfoGain);
        lfoGain.connect(osc2.frequency);

        osc1.connect(masterGain);
        osc2.connect(masterGain);

        osc1.start();
        osc2.start();
        lfo.start();

        audioStarted = true;
    }

    button.addEventListener("click", function() {
        if (!audioStarted) {
            startOminousSound();
        }

        if (isMuted) {
            masterGain.gain.setTargetAtTime(0.12, audioContext.currentTime, 0.2);
            button.innerHTML = "🔊";
            isMuted = false;
        } else {
            masterGain.gain.setTargetAtTime(0.0, audioContext.currentTime, 0.2);
            button.innerHTML = "🔇";
            isMuted = true;
        }
    });
    </script>
    """,
    height=90,
)

df = pd.read_csv("data/spotify_songs_clustered.csv")

saved_model = joblib.load("models/spotify_cluster_model.pkl")

isaved_model = joblib.load("models/spotify_cluster_model.pkl")

if "model" in saved_model:
    model = saved_model["model"]
elif "pipeline" in saved_model:
    model = saved_model["pipeline"]
else:
    st.error("Could not find the model inside spotify_cluster_model.pkl")
    st.stop()

features = saved_model["features"]


st.markdown("<div class='main-title'>Spotify AI Song Cluster</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Choose a song and let the AI glass orb reveal similar tracks.</div>",
    unsafe_allow_html=True
)


song_list = sorted(df["track_name"].dropna().unique())

selected_song = st.selectbox("Choose a song", song_list)

num_results = st.slider("Number of recommendations", 5, 20, 10)

song_row = df[df["track_name"] == selected_song].iloc[0]

cluster_number = song_row["cluster"]
song_name = song_row["track_name"]
artist_name = song_row["track_artist"] if "track_artist" in df.columns else ""
selected_song_url = spotify_search_url(song_name, artist_name)
reaper_img = image_to_base64("assets/reaper_pearl.png")
st.markdown(
    f"""
    <div class="reaper-scene">
        <img src="data:image/png;base64,{reaper_img}" class="reaper-img">
        <div class="pearl-overlay">
            <div class="pearl-song">{song_name}</div>
            <div class="pearl-artist">{artist_name}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Selected Song")

selected_columns = []

for col in ["track_name", "track_artist", "playlist_genre", "playlist_subgenre"]:
    if col in df.columns:
        selected_columns.append(col)

st.dataframe(
    pd.DataFrame([song_row[selected_columns]]),
    use_container_width=True
)
st.link_button(
    f"▶ Play: {song_name} — {artist_name}",
    selected_song_url,
    use_container_width=True
)

same_cluster = df[df["cluster"] == cluster_number].copy()

selected_features = song_row[features].values.reshape(1, -1)
cluster_features = same_cluster[features].values

scaler = model.named_steps["scaler"]

selected_scaled = scaler.transform(selected_features)
cluster_scaled = scaler.transform(cluster_features)

distances = euclidean_distances(selected_scaled, cluster_scaled)[0]

same_cluster["distance"] = distances

recommendations = same_cluster[
    same_cluster["track_name"] != selected_song
].sort_values("distance").head(num_results)


st.subheader("Similar Songs Revealed by the Glass Orb")

recommendation_columns = []

for index, row in recommendations.iterrows():
    rec_song = row["track_name"]
    rec_artist = row["track_artist"] if "track_artist" in recommendations.columns else ""
    rec_url = spotify_search_url(rec_song, rec_artist)

    st.markdown(
        f"""
        <div style="
            padding: 14px;
            margin-bottom: 8px;
            border-radius: 16px;
            background: transparent;
            border: 1px solid rgba(173,216,230,0.25);
        ">
            <div style="font-size: 17px; font-weight: 900;">{rec_song}</div>
            <div style="font-size: 14px; opacity: 0.85;">{rec_artist}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
    f"▶ Play: {rec_song} — {rec_artist}",
    rec_url,
    use_container_width=True
)

st.dataframe(
    recommendations[recommendation_columns],
    use_container_width=True
)




