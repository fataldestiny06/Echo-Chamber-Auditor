"""
Phase 3: Semantic Validator

Role:
    Distinguish potentially harmful ideological echo chambers
    from benign semantic clusters.

Method:
    - Sentiment consensus (VADER)
    - Emotional subjectivity proxy
    - Topic sensitivity gate
"""

from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# ===============================
# CONFIG
# ===============================

SENSITIVE_TOPICS = [
    "politics", "religion", "vaccine", "war",
    "conspiracy", "hate", "extremism"
]

CONSENSUS_THRESHOLD = 0.75
SUBJECTIVITY_THRESHOLD = 0.45
DIRECTION_THRESHOLD = 0.20
ECHO_CONSENSUS_THRESHOLD = 0.85


# ===============================
# HELPERS
# ===============================

def contains_sensitive_topic(texts):
    """
    Checks if cluster discusses sensitive ideological domains.
    """
    full_text = " ".join(texts).lower()
    return any(topic in full_text for topic in SENSITIVE_TOPICS)


def compute_sentiment_metrics(texts, analyzer):
    """
    Computes sentiment consensus and subjectivity proxy
    for a list of texts.
    """
    scores = []

    for text in texts:
        if pd.isna(text):
            continue
        score = analyzer.polarity_scores(str(text))["compound"]
        scores.append(score)

    if not scores:
        return 0.0, 0.0

    # Emotional intensity as subjectivity proxy
    subjectivity = sum(abs(s) for s in scores) / len(scores)

    # Consensus: agreement in sentiment direction
    dominant_direction = 1 if sum(scores) >= 0 else -1
    aligned = [
        s for s in scores
        if (s * dominant_direction) >= DIRECTION_THRESHOLD
    ]

    consensus = len(aligned) / len(scores)

    return consensus, subjectivity


# ===============================
# MAIN VALIDATOR
# ===============================

def run_validation():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "output"

    INPUT_FILE = DATA_DIR / "sbert_dataset.csv"
    OUTPUT_FILE = DATA_DIR / "validated_clusters.csv"

    df = pd.read_csv(INPUT_FILE)

    analyzer = SentimentIntensityAnalyzer()
    results = []

    for cluster_id, group in df.groupby("cluster"):
        titles = group["title"].dropna().tolist()

        consensus, subjectivity = compute_sentiment_metrics(
            titles, analyzer
        )

        sensitive = contains_sensitive_topic(titles)

        # --------------------------
        # LABEL DECISION LOGIC
        # --------------------------
        if (
            sensitive
            and consensus >= CONSENSUS_THRESHOLD
            and subjectivity >= SUBJECTIVITY_THRESHOLD
        ):
            label = "Potential Radicalization Cluster"

        elif consensus >= ECHO_CONSENSUS_THRESHOLD:
            label = "Echo Chamber"

        else:
            label = "Benign Cluster"

        results.append({
            "cluster_id": cluster_id,
            "consensus": round(consensus, 2),
            "subjectivity": round(subjectivity, 2),
            "sensitive_topic": sensitive,
            "label": label
        })

    output_df = pd.DataFrame(results)

    print("\n📊 Cluster Validation Summary:")
    print(output_df)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved validated clusters → {OUTPUT_FILE}")

    return output_df


if __name__ == "__main__":
    run_validation()
