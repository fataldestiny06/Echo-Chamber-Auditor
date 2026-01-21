import pandas as pd
import networkx as nx
from pathlib import Path

CSV_CANDIDATES = [
    Path("main/data/recommendation_walk.csv"),
    Path("recommendation_walk.csv"),
]

OUTPUT_GRAPH = Path("main/output/recommendation_graph.gexf")
OUTPUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)


def load_csv():
    for path in CSV_CANDIDATES:
        if path.exists():
            print(f"📂 Loading recommendation walk CSV: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("❌ recommendation_walk.csv not found in expected locations")


def build_graph():
    df = load_csv()

    G = nx.DiGraph()

    for _, row in df.iterrows():
        src = row.get("source_video")
        dst = row.get("recommended_video")

        if pd.notna(src) and pd.notna(dst):
            G.add_edge(src, dst)

    nx.write_gexf(G, OUTPUT_GRAPH)
    print(f"💾 Graph saved to {OUTPUT_GRAPH}")


if __name__ == "__main__":
    build_graph()
