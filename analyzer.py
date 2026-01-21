import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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
# MAIN
# ===============================

def main():
    print("🔍 Starting Echo Chamber Analysis\n")

    df = load_data()

    if "title" not in df.columns:
        raise ValueError("CSV must contain 'title' column")

    # ---- HANDLE DESCRIPTION COLUMN ----
    if "description" not in df.columns:
        print("⚠️ No 'description' column found — using titles only")
        df["description"] = ""

    # ---- FORCE STRINGS ----
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df["combined_text"] = df["title"] + " [SEP] " + df["description"]
    texts = df["combined_text"].tolist()

    print(f"📄 Total documents: {len(texts)}")
    print(f"📝 Using descriptions: {(df['description'] != '').sum()} / {len(df)} videos")

    # ---- EMBEDDINGS ----
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # ===============================
    # 🔥 NEW: STORE EMBEDDINGS IN CSV
    # ===============================
    emb_dim = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)

    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # ---- CLUSTERING ----
    k = determine_k(len(df))
    print(f"📊 Using k = {k}")

    if k == 1:
        print("⚠️ Too few samples — skipping clustering")
        df["cluster"] = 0
    else:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        df["cluster"] = kmeans.fit_predict(embeddings)

    # ---- METRICS ----
    tightness = compute_cluster_tightness(
        embeddings, df["cluster"].values
    )
    df["cluster_tightness"] = df["cluster"].map(tightness)
    df["potentially_radical"] = (
        df["cluster_tightness"] >= RADICAL_TIGHTNESS_THRESHOLD
    )

    # ---- SAVE DATASET (WITH EMBEDDINGS) ----
    out_csv = OUTPUT_DIR / "sbert_dataset.csv"
    df.to_csv(out_csv, index=False)
    print(f"💾 Saved SBERT dataset with embeddings → {out_csv}")

    # ===============================
    # 🔥 GEOMETRIC EMBEDDING VISUALIZATION
    # ===============================
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
        print(f"🖼 Saved geometric embedding plot → {plot_path}")

    print("\n✅ Analysis complete")
    for c, score in tightness.items():
        if score >= RADICAL_TIGHTNESS_THRESHOLD:
            print(f"⚠️ Potentially radical cluster {c} (tightness={score:.2f})")


if __name__ == "__main__":
    main()
