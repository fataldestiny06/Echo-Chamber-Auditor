import matplotlib
matplotlib.use("Agg")

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

MAIN_DIR = ROOT_DIR / "main"
OUTPUT_DIR = MAIN_DIR / "output"


# Import backend functions
from main.collection.scraper import scrape_recommendations
from main.collection.buildgraph import build_graph
from main.main import run_analysis
from main.Validator.validator import run_validation

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Algorithmic Amplification Auditor",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.metric-box {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.title {
    color: #58a6ff;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)


# -------------------- TITLE --------------------
st.markdown("<div class='title'>Algorithmic Amplification Auditor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Live Audit: YouTube Recommendation Graph </div>",
    unsafe_allow_html=True
)
st.divider()

# -------------------- INPUT --------------------
st.subheader("🔗 Enter YouTube Video URL")
youtube_url = st.text_input("Paste a YouTube link to audit")

run = st.button("Run Algorithmic Audit")

tab1, tab2 = st.tabs(["🏠 Home", "📜 Logs"])

# -------------------- MAIN LOGIC --------------------


if run and youtube_url:
    # Change to main directory for imports
    

    with st.spinner("Phase 1: Scraping YouTube recommendations..."):
        scrape_recommendations(youtube_url, max_steps=20)  # Reduced steps for demo
        st.success("Scraping complete!")

    with st.spinner("Phase 2: Building recommendation graph..."):
        G = build_graph()
        st.success("Graph built!")

    with st.spinner("Phase 3: Running SBERT analysis..."):
        score = run_analysis()
        st.success("Analysis complete!")

    with st.spinner("Phase 4: Validating clusters..."):
        validated_df = run_validation()
        st.success("Validation complete!")

    st.success("Audit complete. Displaying results…")

    # Load outputs
    validated_clusters = pd.read_csv(OUTPUT_DIR / "validated_clusters.csv")

    # Compute metrics
    eci = score["echo_score"]
    rwc = score["intra_similarity"] * 100  # As percentage
    consensus = validated_df["consensus"].mean() * 100  # Average consensus as percentage

    # ==================== HOME TAB ====================
    with tab1:

        # -------------------- METRICS --------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Echo Chamber Index</h3>
                <h1 style="color:#ff4c4c">{eci:.1f}</h1>
                <p>Clicks to Radicalization</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Controversy Score (RWC)</h3>
                <h1 style="color:#ffa657">{rwc:.1f}%</h1>
                <p>Probability of Trap</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Semantic Consensus</h3>
                <h1 style="color:#3fb950">{consensus:.1f}%</h1>
                <p>Subjectivity Level: High</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # -------------------- NETWORK GRAPH --------------------
        st.subheader("🕸 Recommendation Pathways")

        # Load the graph
        graph_file = OUTPUT_DIR / "recommendation_graph.gexf"
        if graph_file.exists():
            G_loaded = nx.read_gexf(graph_file)
        else:
            G_loaded = G  # Use the one from build_graph

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G_loaded, seed=42)

        # Color nodes by step
        node_colors = []
        for node in G_loaded.nodes():
            step = G_loaded.nodes[node].get("step", 0)
            if step < 10:
                node_colors.append("#58a6ff")
            elif step < 20:
                node_colors.append("#ff7b72")
            else:
                node_colors.append("#ff4c4c")

        nx.draw(
            G_loaded, pos, ax=ax2,
            with_labels=False,
            node_color=node_colors,
            node_size=300,
            font_color="white",
            edge_color="#8b949e"
        )

        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")

        st.pyplot(fig2)

        # ==================== LOGS TAB ====================
        with tab2:

            st.subheader("📜 Phase 3: Semantic Validator Logs")

            st.dataframe(
                validated_clusters,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("""
✅ **Pipeline completed successfully**  
📁 Output files saved in `output/` folder
""")

    
else:
    st.info("Enter a YouTube link and click **Run Algorithmic Audit** to begin.")
