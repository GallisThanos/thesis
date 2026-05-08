import os
import numpy as np

from config import Config
from pipeline import extract_features_from_file, average_features, std_features, interpret

config = Config(
    sr=44100,
    avg_method="trimmed",
    momentary_window=0.4,
    trim_db=40,
)


def process_dataset(dataset_path: str) -> dict:
    genres = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    results = {}

    for genre in genres:
        genre_path   = os.path.join(dataset_path, genre)
        feature_list = []
        print(f"\nProcessing: {genre}")

        for root, _, files in os.walk(genre_path):
            for file in files:
                if file.lower().endswith(".wav"):
                    path = os.path.join(root, file)
                    print(f"  -> {file}")
                    try:
                        features = extract_features_from_file(path, config)
                        feature_list.append(features)
                    except Exception as e:
                        print(f"  Error on {file}: {e}")

        if feature_list:
            avg = average_features(feature_list, config)
            std = std_features(feature_list)
            results[genre] = {"avg": avg, "std": std, "n": len(feature_list)}
        else:
            print(f"  No WAV files found in {genre}")

    return results


def main():
    dataset_path = r"C:\Users\fazet\Desktop\RWC DATASET"

    if not os.path.exists(dataset_path):
        print("Dataset path not found.")
        return

    results = process_dataset(dataset_path)

    print("\n\n=== FINAL AVERAGE FEATURES ===")
    for genre, data in results.items():
        print(f"\nGenre: {genre}  (n={data['n']})")
        for key in data["avg"]:
            print(f"  {key}: avg={data['avg'][key]:.4f}  std={data['std'][key]:.4f}")


if __name__ == "__main__":
    main()
