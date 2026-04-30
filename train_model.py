import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


DATA_PATH = "data/spotify_songs.csv"
MODEL_PATH = "models/spotify_cluster_model.pkl"
OUTPUT_DATA_PATH = "data/spotify_songs_clustered.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded.")
    print("Shape:", df.shape)
    return df


def prepare_features(df):
    numeric_features = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    available_features = [col for col in numeric_features if col in df.columns]

    if len(available_features) == 0:
        raise ValueError("No expected audio feature columns found.")

    print("Using features:")
    print(available_features)

    model_df = df.dropna(subset=available_features).copy()
    X = model_df[available_features]

    return model_df, X, available_features


def test_number_of_clusters(X_train_scaled):
    cluster_options = [5, 8, 10, 12, 15]
    results = []

    print("\nTesting cluster counts...")

    for k in cluster_options:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_train_scaled)

        score = silhouette_score(X_train_scaled, labels)
        results.append((k, score))

        print(f"k={k}, silhouette score={score:.4f}")

    best_k = max(results, key=lambda item: item[1])[0]
    print(f"\nBest k: {best_k}")

    return best_k


def main():
    df = load_data()
    model_df, X, features = prepare_features(df)

    X_train, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_k = test_number_of_clusters(X_train_scaled)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10))
    ])

    pipeline.fit(X)

    train_labels = pipeline.predict(X_train)
    test_labels = pipeline.predict(X_test)

    X_train_final_scaled = pipeline.named_steps["scaler"].transform(X_train)
    X_test_final_scaled = pipeline.named_steps["scaler"].transform(X_test)

    train_score = silhouette_score(X_train_final_scaled, train_labels)
    test_score = silhouette_score(X_test_final_scaled, test_labels)

    print("\nFinal model test:")
    print(f"Train silhouette score: {train_score:.4f}")
    print(f"Test silhouette score: {test_score:.4f}")

    model_df["cluster"] = pipeline.predict(X)

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": features
        },
        MODEL_PATH
    )

    model_df.to_csv(OUTPUT_DATA_PATH, index=False)

    print("\nModel saved to:", MODEL_PATH)
    print("Clustered dataset saved to:", OUTPUT_DATA_PATH)

    print("\nCluster counts:")
    print(model_df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
