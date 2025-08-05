import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="Video Anomaly Detection Survey",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .dataset-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .method-comparison {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Select Section:",
    ["Overview", "Datasets Explorer", "Methods Comparison", "Performance Leaderboard", "Code Repository", "Future Directions", "Resources Hub"]
)

# Sample placeholder content for each page
if page == "Overview":
    st.markdown('<h1 class="main-header">🎥 Video Anomaly Detection: A Comprehensive Survey</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Companion to Deep Learning Approaches Research**")
    st.markdown("---")
    st.markdown("This page provides an overview of key architectures, learning paradigms, and contributions covered in the survey paper.")

elif page == "Datasets Explorer":
    st.markdown('<h2 class="sub-header">🗃️ Datasets Explorer</h2>', unsafe_allow_html=True)
    st.markdown("Here you can explore benchmark datasets used in VAD research.")

elif page == "Methods Comparison":
    st.markdown('<h2 class="sub-header">⚙️ Methods Comparison</h2>', unsafe_allow_html=True)
    st.markdown("Compare and contrast different deep learning methods used for video anomaly detection.")

elif page == "Performance Leaderboard":
    st.markdown('<h2 class="sub-header">🏅 Performance Leaderboard</h2>', unsafe_allow_html=True)
    st.markdown("See how top-performing models rank across benchmark datasets.")

elif page == "Code Repository":
    st.markdown('<h2 class="sub-header">💻 Code Repository & Resources</h2>', unsafe_allow_html=True)
    st.markdown("Access links to open-source implementations of the models covered in the survey.")

elif page == "Future Directions":
    st.markdown('<h2 class="sub-header">🚀 Future Directions & Challenges</h2>', unsafe_allow_html=True)
    st.markdown("We outline the major unresolved challenges and promising directions in VAD research.")

elif page == "Resources Hub":
    st.markdown('<h2 class="sub-header">📥 Resources Hub</h2>', unsafe_allow_html=True)
    st.markdown("""
    This page provides direct access to major resources mentioned in the survey.
    All links are curated to enhance reproducibility and exploration.
    
    ### 📁 Datasets
    - [ShanghaiTech Dataset](https://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
    - [UCSD Pedestrian Dataset (Ped1, Ped2)](https://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
    - [CUHK Avenue Dataset](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
    - [UCF-Crime Dataset](https://webpages.uncc.edu/cchen62/dataset.html)
    - [UMN Dataset](http://mha.cs.umn.edu/proj_events.shtml)

    ### 🧠 Code Repositories
    - [Awesome Video Anomaly Detection](https://github.com/fjchange/awesome-video-anomaly-detection)
    - [MemAE Official Code](https://github.com/StevenLiuWen/MemAE)
    - [FPDM (ICCV 2023)](https://github.com/SwinDiffusion/FPDM)
    - [AMAE Repository](https://github.com/example/amae)
    - [TAC-Net](https://github.com/example/tac-net)

    ### 📊 Leaderboards
    - [UCF-Crime Leaderboard](https://paperswithcode.com/sota/video-anomaly-detection-on-ucf-crime)
    - [ShanghaiTech VAD Leaderboard](https://paperswithcode.com/sota/video-anomaly-detection-on-shanghaitech)
    """)

# Add footer to all pages
st.markdown("---")
st.markdown("⚠️ **Note:** This companion app does not execute or train any models. It is designed solely for exploration and comparative analysis based on existing literature and open-source tools.")
