
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    print("\n========== analytIQ ML TRAINING ==========\n")

    # Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["review_text", "sentiment"]
    )

    df["review_text"] = df["review_text"].astype(str)
    df["sentiment"] = (
        df["sentiment"].astype(str).str.lower().str.strip()
    )

    X = df["review_text"]
    y = df["sentiment"]

    print("Total training examples:", len(df))
    print("Sentiment classes:", y.unique())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Train
    print("\nTraining Logistic Regression model...")

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== MODEL EVALUATION ==========")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)

    print("\n========== TRAINING COMPLETE ==========")


if __name__ == "__main__":
    train_model()
