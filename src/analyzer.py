
import joblib
import re

from pathlib import Path
from collections import Counter

# Find project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Model location
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"

# Verify model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ML model not found at {MODEL_PATH}. "
        "Run python train_model.py first."
    )

# Load trained ML model
model = joblib.load(MODEL_PATH)

print("analytIQ ML model loaded successfully!")


# Aspect dictionary
ASPECTS = {
    "Battery": [
        "battery", "backup", "charging", "charge"
    ],

    "Camera": [
        "camera", "photo", "picture", "video"
    ],

    "Display": [
        "display", "screen", "brightness"
    ],

    "Audio": [
        "sound", "speaker", "audio", "microphone"
    ],

    "Build Quality": [
        "build", "material", "plastic",
        "metal", "durable"
    ],

    "Price": [
        "price", "expensive", "affordable",
        "cost", "overpriced"
    ],

    "Delivery": [
        "delivery", "shipping",
        "packaging", "package"
    ],

    "Design": [
        "design", "appearance", "style", "look"
    ],

    "Performance": [
        "performance", "speed", "fast",
        "slow", "smooth"
    ],

    "Customer Service": [
        "service", "support", "customer care"
    ],

    "Comfort": [
        "comfort", "comfortable", "fit", "fabric"
    ],

    "Usability": [
        "easy to use", "difficult to use",
        "convenient", "user friendly"
    ]
}


# Split review into sentences
def split_sentences(text):

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            str(text)
        )
        if len(sentence.strip()) > 3
    ]


# Extract aspects from sentence
def extract_aspects(sentence):

    sentence_lower = sentence.lower()

    found = []

    for aspect, keywords in ASPECTS.items():

        for keyword in keywords:

            pattern = (
                r"\b" + re.escape(keyword) + r"\b"
            )

            if re.search(pattern, sentence_lower):
                found.append(aspect)
                break

    return found


# Analyze one review
def analyze_review(review):

    sentences = split_sentences(review)

    results = []

    for sentence in sentences:

        aspects = extract_aspects(sentence)

        if not aspects:
            continue

        probabilities = model.predict_proba([sentence])[0]

        classes = model.classes_

        index = probabilities.argmax()

        sentiment = classes[index]

        confidence = float(probabilities[index])

        for aspect in aspects:

            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "sentence": sentence
            })

    # Fallback when no aspect is identified
    if not results and str(review).strip():

        probabilities = model.predict_proba(
            [str(review)]
        )[0]

        index = probabilities.argmax()

        results.append({
            "aspect": "Overall",
            "sentiment": model.classes_[index],
            "confidence": float(probabilities[index]),
            "sentence": str(review)
        })

    return results


# Analyze multiple reviews
def analyze_reviews(reviews):

    all_results = []

    for review_id, review in enumerate(reviews):

        results = analyze_review(review)

        for result in results:

            result["review_id"] = review_id

            all_results.append(result)

    return all_results


# Generate feedback summary
def generate_summary(results):

    if not results:
        return "No usable feedback was found."

    aspect_counts = {}

    for item in results:

        aspect = item["aspect"]
        sentiment = item["sentiment"]

        if aspect not in aspect_counts:
            aspect_counts[aspect] = Counter()

        aspect_counts[aspect][sentiment] += 1

    positive = sum(
        1 for item in results
        if item["sentiment"] == "positive"
    )

    negative = sum(
        1 for item in results
        if item["sentiment"] == "negative"
    )

    neutral = sum(
        1 for item in results
        if item["sentiment"] == "neutral"
    )

    strengths = []
    weaknesses = []

    for aspect, counts in aspect_counts.items():

        if counts["positive"] > counts["negative"]:
            strengths.append(aspect)

        elif counts["negative"] > counts["positive"]:
            weaknesses.append(aspect)

    summary = [
        f"Analyzed {len(results)} aspect-level feedback observations.",
        (
            f"Positive: {positive}, "
            f"Negative: {negative}, "
            f"Neutral: {neutral}."
        )
    ]

    if strengths:

        summary.append(
            "Customers generally appreciate: "
            + ", ".join(strengths) + "."
        )

    if weaknesses:

        summary.append(
            "Areas needing improvement: "
            + ", ".join(weaknesses) + "."
        )

    return " ".join(summary)
