import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics.pairwise import euclidean_distances


MODEL_PATH = "models/spotify_cluster_model.pkl"
DATA_PATH = "data/spotify_songs_clustered.csv"


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["features"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def get_display_columns(df):
    possible_columns = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "cluster"
    ]

    return [col for col in possible_columns if col in df.columns]


def recommend_similar_songs(selected_song_name, df, pipeline, features, number_of_recommendations):
    selected_rows = df[df["track_name"] == selected_song_name]

    if selected_rows.empty:
        return pd.DataFrame()

    selected_song = selected_rows.iloc[0]
    selected_cluster = selected_song["cluster"]

    same_cluster = df[df["cluster"] == selected_cluster].copy()

    selected_features = selected_song[features].values.reshape(1, -1)
    cluster_features = same_cluster[features].values

    scaler = pipeline.named_steps["scaler"]

    selected_scaled = scaler.transform(selected_features)
    cluster_scaled = scaler.transform(cluster_features)

    distances = euclidean_distances(selected_scaled, cluster_scaled)[0]

    same_cluster["similarity_distance"] = distances

    recommendations = same_cluster[
        same_cluster["track_name"] != selected_song_name
    ].sort_values("similarity_distance").head(number_of_recommendations)

    return recommendations


def main():
    st.set_page_config(
        page_title="Spotify Song Clustering",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 Spotify Song Clustering App")
    st.write("Find songs with similar audio attributes using clustering.")

    pipeline, features = load_model()
    df = load_data()

    st.sidebar.header("Settings")

    number_of_recommendations = st.sidebar.slider(
        "Number of similar songs",
        min_value=5,
        max_value=30,
        value=10
    )

    song_names = sorted(df["track_name"].dropna().unique())

    selected_song = st.selectbox(
        "Choose a song:",
        song_names
    )

    selected_song_rows = df[df["track_name"] == selected_song]

    if not selected_song_rows.empty:
        selected_song_info = selected_song_rows.iloc[0]

        st.subheader("Selected Song")
        display_columns = get_display_columns(df)

        st.dataframe(
            selected_song_rows[display_columns].head(1),
            use_container_width=True
        )

        cluster_id = selected_song_info["cluster"]
        st.info(f"This song belongs to cluster {cluster_id}.")

        recommendations = recommend_similar_songs(
            selected_song,
            df,
            pipeline,
            features,
            number_of_recommendations
        )

        st.subheader("Recommended Similar Songs")

        if recommendations.empty:
            st.warning("No similar songs found.")
        else:
            display_columns = get_display_columns(recommendations)
            display_columns.append("similarity_distance")

            st.dataframe(
                recommendations[display_columns],
                use_container_width=True
            )

    st.subheader("Cluster Overview")
    cluster_counts = df["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)


if __name__ == "__main__":
    main()
