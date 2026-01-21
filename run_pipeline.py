from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

SCRAPER = BASE_DIR / "main" / "collection" / "scraper.py"
GRAPH_BUILDER = BASE_DIR / "main" / "collection" / "buildgraph.py"
ANALYZER = BASE_DIR / "analyzer.py"

print("▶ Phase 1: Running scraper...")
subprocess.run([sys.executable, SCRAPER], check=True)

print("▶ Phase 2: Building graph...")
subprocess.run([sys.executable, GRAPH_BUILDER], check=True)

print("▶ Phase 3: Running analyzer...")
subprocess.run([sys.executable, ANALYZER], check=True)

print("✅ Pipeline completed successfully")
