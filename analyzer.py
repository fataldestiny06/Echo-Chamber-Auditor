import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ===============================
# CONFIG
# ===============================

DATA_CANDIDATES = [
    Path("main/data/recommendation_walk.csv"),
    Path("recommendation_walk.csv"),
    Path("output/recommendation_walk.csv"),
]

OUTPUT_DIR = Path("main/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
RADICAL_TIGHTNESS_THRESHOLD = 0.65


# ===============================
# HELPERS
# ===============================

def load_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            print(f"📂 Loaded data from: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("❌ recommendation_walk.csv not found")


def determine_k(n):
    if n < 3:
        return 1
    return min(8, max(3, int(np.sqrt(n))))


def compute_cluster_tightness(embeddings, labels):
    scores = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            scores[c] = 0.0
        else:
            sims = np.dot(embeddings[idx], embeddings[idx].T)
            scores[c] = sims.mean()
    return scores


# ===============================
# CORE ANALYSIS FUNCTION
# ===============================

def run_analysis():
    print("🔍 Starting Echo Chamber Analysis\n")

    df = load_data()

    if "title" not in df.columns:
        raise ValueError("CSV must contain 'title' column")

    if "description" not in df.columns:
        df["description"] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df["combined_text"] = df["title"] + " [SEP] " + df["description"]
    texts = df["combined_text"].tolist()

    print(f"📄 Total documents: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # ---- STORE EMBEDDINGS ----
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # ---- CLUSTERING ----
    k = determine_k(len(df))
    if k == 1:
        df["cluster"] = 0
    else:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        df["cluster"] = kmeans.fit_predict(embeddings)

    # ---- METRICS ----
    tightness = compute_cluster_tightness(
        embeddings, df["cluster"].values
    )
    df["cluster_tightness"] = df["cluster"].map(tightness)
    df["potentially_radical"] = df["cluster_tightness"] >= RADICAL_TIGHTNESS_THRESHOLD

    # ---- SAVE DATASET ----
    out_csv = OUTPUT_DIR / "sbert_dataset.csv"
    df.to_csv(out_csv, index=False)

    # ---- PCA VIS ----
    if len(df) > 1:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=df["cluster"],
            cmap="tab10",
            alpha=0.7
        )
        plt.title("Geometric SBERT Embeddings (PCA)")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.colorbar(label="Cluster")

        plot_path = OUTPUT_DIR / "clusters_2d.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # ---- RETURN METRICS FOR APP ----
    return {
        "echo_score": max(tightness.values()) if tightness else 0.0,
        "intra_similarity": np.mean(list(tightness.values())) if tightness else 0.0
    }


# ===============================
# 🔥 COMPATIBILITY CLASS (FIXES IMPORT ERROR)
# ===============================

class EchoChamberAnalyzer:
    def __init__(self):
        pass

    def run(self):
        return run_analysis()


# ===============================
# CLI ENTRY POINT
# ===============================

if __name__ == "__main__":
    run_analysis()
