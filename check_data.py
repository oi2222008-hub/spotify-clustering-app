import pandas as pd

df = pd.read_csv("data/spotify_songs.csv")

print("Shape:", df.shape)
print(df.columns.tolist())
print(df.head())
