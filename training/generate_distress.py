import pandas as pd
import numpy as np
from pathlib import Path


INPUT = "data/processed/emotion_features.csv"
OUTPUT = "data/processed/distress_features.csv"


def calculate_distress_score(row):

    # Autism emotion probabilities
    # Order:
    # Natural, anger, fear, joy, sadness, surprise

    autism_negative = (
        row["autism_anger"]
        + row["autism_fear"]
        + row["autism_sadness"]
    )

    # General FER probabilities
    # angry, disgust, fear, happy, neutral, sad, surprise

    general_negative = (
        row["general_angry"]
        + row["general_disgust"]
        + row["general_fear"]
        + row["general_sad"]
    )

    # Behavioral features
    movement = row["movement"]
    head_movement = row["head_movement"]

    # Initial distress score
    score = (
        0.35 * autism_negative
        + 0.25 * general_negative
        + 0.25 * movement
        + 0.15 * head_movement
    )

    return score


def main():

    df = pd.read_csv(INPUT)

    df["distress_score"] = df.apply(
        calculate_distress_score,
        axis=1
    )

    # Initial threshold
    threshold = df["distress_score"].median()

    df["distress_label"] = (
        df["distress_score"] >= threshold
    ).astype(int)

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("==============================")
    print("DISTRESS DATA GENERATED")
    print("==============================")

    print("Samples:", len(df))
    print("Threshold:", threshold)

    print("\nLabels:")
    print(df["distress_label"].value_counts())

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()