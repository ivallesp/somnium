# Spotify Tracks

[Kaggle dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) with 114k tracks across 114 genres. Uses 11 continuous audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, and popularity. A random subsample of 10,000 tracks is used to train a 20x20 hexagonal SOM.

## Run

```bash
uv run python examples/spotify/run.py
```

Data is downloaded automatically on first run via `prepare.py`.
