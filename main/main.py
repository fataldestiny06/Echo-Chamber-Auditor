import os
import sys

# Add project root to sys.path for absolute imports
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# HARD DISABLE THREADING (important for macOS + PyTorch)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from analyzer import EchoChamberAnalyzer
from pathlib import Path

def run_analysis():
    print("=" * 70)
    print("ECHO CHAMBER & ALGORITHMIC AMPLIFICATION AUDITOR")
    print("=" * 70, "\n")

    analyzer = EchoChamberAnalyzer()

    # Load REAL YouTube recommendation data
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "output"
    OUTPUT_DIR = DATA_DIR
    input_file = DATA_DIR / "recommendation_walk.csv"
    if not input_file.exists():
        raise FileNotFoundError(
            "recommendation_walk.csv not found. Please run scraper first."
        )

    analyzer.load_data(input_file)
    analyzer.load_data(DATA_DIR / "recommendation_walk.csv")
    # SBERT pipeline
    analyzer.create_embeddings()
    analyzer.cluster_videos(n_clusters=5)

    # Geometry
    analyzer.visualize_clusters(OUTPUT_DIR / "clusters_2d.png")
    analyzer.generate_sbert_dataset(OUTPUT_DIR / "sbert_dataset.csv")
    # Echo score for entire walk (simulated user)
    user_indices = list(range(len(analyzer.df)))
    score = analyzer.calculate_user_echo_score(user_indices)

    print("USER ECHO CHAMBER REPORT")
    print(score)
    print("\nPIPELINE COMPLETE")

    return score

if __name__ == "__main__":
    run_analysis()
