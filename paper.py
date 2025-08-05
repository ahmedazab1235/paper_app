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

# Sidebar navigation using radio buttons (bar style)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Datasets Explorer", "Methods Comparison", "Code Repository", "Future Directions"]
)

# Sample placeholder content for each page
if page == "Overview":
    st.markdown('<h1 class="main-header">🎥 Video Anomaly Detection: A Comprehensive Survey of Deep Learning Approaches</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Companion to Deep Learning Approaches Research**")
    st.markdown("---")
    st.markdown("Video anomaly detection (VAD) plays a crucial role in intelligent surveillance systems, aiming to identify abnormal events that deviate from usual patterns in real-world environments. With the proliferation of deep learning technologies, significant progress has been made using unsupervised and weakly supervised learning approaches that address the scarcity of labeled anomalous data. This survey provides a comprehensive overview of the recent advancements in VAD, categorizing methods based on reconstruction and prediction paradigms and highlighting key innovations in generative models such as autoencoders, variational autoencoders (VAEs), and generative adversarial networks (GANs). We further explore advanced spatiotemporal learning techniques, attention mechanisms, and multi-stream architectures that capture complex temporal dependencies and spatial relationships. A detailed review of benchmark datasets and evaluation metrics is presented, followed by a comparative analysis of state-of-the-art methods across different scenarios. Finally, we discuss open challenges including real-time processing, cross-domain generalization, and interpretability, along with future research directions to guide continued advancement in this rapidly evolving domain.")

elif page == "Datasets Explorer":
    st.markdown('<h2 class="sub-header">🗃️ Datasets Explorer</h2>', unsafe_allow_html=True)

    datasets = [
        {
            "name": "UCSD Pedestrian Dataset (Ped1)",
            "description": "UCSD Pedestrian dataset includes two subsets (Ped1) designed for video anomaly detection in pedestrian walkways. Ped1 contains 34 grayscale sequences (12,880 frames). Anomalies involve bikers, skaters, carts, and pedestrians walking against flow.",
            "key_features": [
                "**Total Sequences:** 34",
                "**Raw Image Size:** 238 × 158",
                "**Processed Image Size:** 227 × 227",
                "**Training Set Size:** 34 clips",
                "**Testing Set Size:** 36 clips",
                "**Anomalies:** Biking, skateboarding, riding carts, walking in wrong direction"
            ],
            "video_url": "https://www.youtube.com/watch?v=7PLlImkk3Wg",
            "download_link": "https://www.kaggle.com/datasets/aryashah2k/ucsd-pedestrian-database"
        },
        {
            "name": "UCSD Pedestrian Dataset (Ped2)",
            "description": "UCSD Pedestrian dataset includes two subsets (Ped2) designed for video anomaly detection in pedestrian walkways. Ped2 includes 16 sequences (4,200 frames). Anomalies involve bikers, skaters, carts, and pedestrians walking against flow.",
            "key_features": [
                "**Total Sequences:** 16",
                "**Raw Image Size:** 360 × 240",
                "**Processed Image Size:** 227 × 227",
                "**Training Set Size:** 16 clips",
                "**Testing Set Size:** 12 clips",
                "**Anomalies:** Biking, skateboarding, riding carts, walking in wrong direction"
            ],
            "video_url": "https://www.youtube.com/watch?v=kDKP8oGLaRg",
            "download_link": "https://paperswithcode.com/dataset/ucsd"
        },
        {
            "name": "CUHK Avenue",
            "description": "CUHK Avenue dataset is captured from an urban university campus and includes 37 color video clips. It contains various abnormal activities such as throwing objects, loitering, and running. The background is relatively consistent, aiding detection in fixed-camera scenarios.",
            "key_features": [
                "**Total Videos:** 37",
                "**Raw Image Size:** 640 × 360",
                "**Processed Image Size:** 224 × 224",
                "**Training Set Size:** 16 clips",
                "**Testing Set Size:** 21 clips",
                "**Anomalies:** Throwing bags, running, loitering"
            ],
            "video_url": "https://www.youtube.com/watch?v=NqUSzeeCp-g",
            "download_link": "https://www.kaggle.com/datasets/vibhavvasudevan/avenue"
        },
        {
            "name": "ShanghaiTech",
            "description": "ShanghaiTech dataset includes 437 videos across 13 real-world scenes with diverse environments. It contains over 130 abnormal events including loitering, fighting, and sudden running. It is known for its complexity and scene diversity.",
            "key_features": [
                "**Total Videos:** 437",
                "**Raw Image Size:** Varies",
                "**Processed Image Size:** 224 × 224",
                "**Training Set Size:** 330 videos",
                "**Testing Set Size:** 107 videos",
                "**Scenes:** 13 real-world environments",
                "**Anomalies:** Loitering, fighting, riding, sudden running"
            ],
            "video_url": "https://www.youtube.com/watch?v=u2BpTYt5umA",
            "download_link": "https://svip-lab.github.io/dataset/campus_dataset.html"
        },
        {
            "name": "UCF-Crime",
            "description": "UCF-Crime is one of the largest real-world surveillance datasets with 1,900 untrimmed videos and 13 anomaly types. It includes long-range events like robbery, shooting, and accidents, making it suitable for weakly supervised training.",
            "key_features": [
                "**Total Videos:** 1,900",
                "**Resolution:** 320 × 240 (avg)",
                "**Anomalies:** Robbery, shooting, fighting, accident, vandalism"
            ],
            "video_url": "https://www.youtube.com/watch?v=8TKkPePFpiE&rco=1",
            "download_link": "https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset"
        },
        {
            "name": "UMN",
            "description": "UMN dataset is a simulation-based video anomaly dataset recorded in indoor and outdoor settings. It consists of 11 grayscale videos portraying crowds switching from normal to panic behavior.",
            "key_features": [
                "**Total Videos:** 11",
                "**Resolution:** 320 × 240",
                "**Scenes:** Indoor and outdoor",
                "**Anomalies:** Sudden running, panic crowd behavior"
            ],
            "video_url": "https://www.youtube.com/watch?v=fq_EXM9Zsvg",
            "download_link": "http://mha.cs.umn.edu/proj_events.shtml"
        }
    ]

    for data in datasets:
        with st.expander(f"📁 {data['name']}"):
            st.markdown(data["description"])
            st.markdown("### Key Features:")
            for feature in data["key_features"]:
                st.markdown(f"- {feature}")
            st.markdown("### Example Video:")
            st.video(data["video_url"])
            st.markdown("### Download Link:")
            st.markdown(f"[Click here to download]({data['download_link']})")

elif page == "Methods Comparison":
    st.markdown('<h2 class="sub-header">⚙️ Methods Comparison</h2>', unsafe_allow_html=True)
    st.markdown("Compare and contrast different deep learning methods used for video anomaly detection.")

    st.markdown("### 📊 Performance Comparison Table")
    methods_table = '''
| Method | Ped1 AUC (%) | Ped1 EER (%) | Ped2 AUC (%) | Ped2 EER (%) | Avenue AUC (%) | Avenue EER (%) | ShanghaiTech AUC (%) | ShanghaiTech EER (%) |
|--------|--------------|--------------|--------------|--------------|----------------|----------------|-----------------------|----------------------|
| Conv-AE (Hasan et al., 2016) [12] | 81.0 | 27.9 | 90.0 | 21.7 | 70.2 | 25.1 | 60.9 | 42.0 |
| Stacked RNN + Sparse Coding (Luo et al., 2017) [22] | — | — | 92.21 | — | 81.71 | — | 68.00 | — |
| MemAE (Gong et al., 2019) [14] | — | — | 94.1 | — | 83.3 | — | 71.2 | — |
| MNAD-Reconstruction (Park et al., 2020) [15] | — | — | 90.2 | — | 82.8 | — | 69.8 | — |
| NM-GAN – Noise-Modulated GAN (Chen et al., 2021) [28] | 90.7 | 15.0 | 96.3 | 6.0 | 88.6 | 15.3 | 85.3 | 17.0 |
| CycleGAN with Skeleton Features (Fan et al., 2022) [29] | — | — | — | — | 87.8 | — | — | — |
| AMAE – Appearance-Motion AE (Liu et al., 2022) [30] | — | — | 97.4 | — | 88.2 | — | 73.6 | — |
| FPDM – Feature Prediction Diffusion (Yan et al., 2023) [31] | — | — | — | — | 90.1 | — | 78.6 | — |
    '''
    st.markdown(methods_table)

    st.markdown("### 📈 Average AUC Comparison (Where Available)")
    data = {
        "Method": ["Conv-AE", "Stacked RNN", "MemAE", "MNAD", "NM-GAN", "CycleGAN", "AMAE", "FPDM"],
        "Average AUC": [75.5, 80.64, 82.87, 80.93, 90.23, 87.8, 86.4, 84.35]
    }
    df_auc = pd.DataFrame(data)
    fig_auc = px.bar(df_auc, x="Method", y="Average AUC", color="Method",
                     title="Average AUC Across Datasets (Non-missing values only)",
                     labels={"Average AUC": "AUC (%)"})
    st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("### ⚡ Performance vs Error Rate")
    st.markdown("Visualizing how high-performing models trade off with their Equal Error Rate (EER).")

    eer_data = {
        "Method": ["Conv-AE", "NM-GAN"],
        "Ped2 AUC": [90.0, 96.3],
        "Ped2 EER": [21.7, 6.0]
    }
    eer_df = pd.DataFrame(eer_data)
    fig_scatter = px.scatter(eer_df, x="Ped2 AUC", y="Ped2 EER", text="Method",
                             title="AUC vs EER on Ped2 Dataset",
                             labels={"Ped2 AUC": "AUC (%)", "Ped2 EER": "EER (%)"})
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Code Repository":
    st.markdown('<h2 class="sub-header">💻 Code Repository & Resources</h2>', unsafe_allow_html=True)
    st.markdown("Access links to open-source implementations of the models covered in the survey.")

    repo_data = [
        {"name": "Conv-AE (Hasan et al., 2016)", "url": "https://github.com/ShrishtiHore/Anomaly-Detection-in-CCTV-Surveillance-Videos/blob/master/convAE%20(1).ipynb"},
        {"name": "Stacked RNN + Sparse Coding (Luo et al., 2017)", "url": "https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection"},
        {"name": "MemAE (Gong et al., 2019)", "url": "https://github.com/donggong1/memae-anomaly-detection"},
        {"name": "MNAD (Park et al., 2020)", "url": "https://github.com/cvlab-yonsei/MNAD"},
        {"name": "MDPS: Unsupervised anomaly detection via masked diffusion posterior sampling Wu et al. Arxiv '24.", "url": "https://github.com/KevinBHLin/MDPS"},
        {"name": "Unsupervised industrial anomaly detection with diffusion models Xu et al. J. Vis. Commun. Image R. '23.", "url": "https://github.com/xhh12381/RecDMs-AD"},
        {"name": "DiffAD: Imputation-based time-series anomaly detection with conditional weight-incremental diffusion models Xiao et al. KDD '23.", "url": "https://github.com/ChunjingXiao/DiffAD"},
        {"name": "LDM: Unsupervised 3D out-of-distribution detection with latent diffusion models Graham et al. MICCAI '23.", "url": "https://github.com/marksgraham/ddpm-ood"},
        {"name": "AnoDDPM: AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise Wyatt et al. CVPR '22.", "url": "https://github.com/Julian-Wyatt/AnoDDPM"},
        {"name": "DTE: On diffusion modeling for anomaly detection Livernoche et al. ICLR '24.", "url": "https://github.com/vicliv/dte"},
        {"name": "Progressive distillation for fast sampling of diffusion models Salimans et al. ICLR '22.", "url": "https://github.com/google-research/google-research/tree/master/diffusion_distillation"},
        {"name": "GLAD: GLAD: Towards better reconstruction with global and local adaptive diffusion models for unsupervised anomaly detection Yao et al. Arxiv '24.", "url": "https://github.com/hyao1/GLAD"},
        {"name": "MDPS: Unsupervised anomaly detection via masked diffusion posterior sampling Wu et al. Arxiv '24.", "url": "https://github.com/KevinBHLin/MDPS"},
        {"name": "DualAnoDiff: DualAnoDiff: Dual-interrelated diffusion model for few-shot anomaly image generation Jin et al. arXiv '24.", "url": "https://github.com/yinyjin/DualAnoDiff"},
        {"name": "AnomalyXFusion: AnomalyXFusion: Multi-modal anomaly synthesis with diffusion Hu et al. Arxiv '24.", "url": "https://github.com/hujiecpp/MVTec-Caption"},
        {"name": "IgCONDA-PET: IgCONDA-PET: Implicitly-guided counterfactual diffusion for detecting anomalies in PET images Ahamed et al. Arxiv '24.", "url": "https://github.com/igcondapet/IgCONDA-PET"},
        {"name": "MMCCD: Modality cycles with masked conditional diffusion for unsupervised anomaly segmentation in MRI Liang et al. Arxiv '23.", "url": "https://github.com/ZiyunLiang/MMCCD"},
        {"name": "mDDPM: Unsupervised anomaly detection in medical images using masked diffusion model Iqbal et al. MLMI '23.", "url": "https://mddpm.github.io/"},
        {"name": "THOR: Diffusion models with implicit guidance for medical anomaly detection Linguraru et al. Arxiv '24.", "url": "https://github.com/ci-ber/THOR_DDPM"},
        {"name": "Masked Bernoulli Diffusion: Binary noise for binary tasks: Masked bernoulli diffusion for unsupervised anomaly detection Linguraru et al. arXiv '24.", "url": "https://github.com/JuliaWolleb/Anomaly_berdiff"},
        {"name": "Unsupervised industrial anomaly detection with diffusion models Xu et al. J. Vis. Commun. Image R. '23.", "url": "https://github.com/xhh12381/RecDMs-AD"},
        {"name": "Collaborative Diffusion: Collaborative diffusion for multi-modal face generation and editing Huang et al. CVPR '23.", "url": "https://github.com/ziqihuangg/Collaborative-Diffusion"},
        {"name": "Diffusion models for medical anomaly detection Wolleb et al. MICCAI '22. ", "url": "https://gitlab.com/cian.unibas.ch/diffusion-anomaly"},
        {"name": "PHAD: Prototype-oriented hypergraph representation learning for anomaly detection in tabular data Li et al. Information Processing and Management '25.", "url": "https://github.com/ls-xju/pro_ad"},
        {"name": "more ..", "url": "https://github.com/fdjingliu/DMAD?tab=readme-ov-file"},
    ]

    for repo in repo_data:
        st.markdown(f"- [{repo['name']}]({repo['url']})")


elif page == "Future Directions":
    st.markdown('<h2 class="sub-header">🚀 Future Directions & Challenges</h2>', unsafe_allow_html=True)
    st.markdown("""
Despite significant progress in video anomaly detection (VAD), several critical challenges remain unresolved. Addressing these gaps is essential to improve the reliability, adaptability, and deployment of VAD systems in real-world scenarios. This section outlines key limitations and emerging research directions:

### 🔄 Generalization Across Scenes and Domains
Models trained on one dataset often fail in unseen environments due to differences in lighting, scene layout, camera angle, and anomaly types. Overfitting and data bias are particularly problematic for supervised and weakly supervised approaches.
- **Future directions:** Domain adaptation techniques, Zero-shot and few-shot anomaly detection, Cross-domain pretraining.

### ⚡ Real-Time Processing and Scalability
Most VAD models are computationally heavy, limiting their use in edge devices or large-scale deployments.
- **Challenges:** High inference latency, memory usage, and model size.
- **Future directions:** Lightweight architectures, Knowledge distillation, Event-triggered processing mechanisms.

### 🔍 Interpretability of Deep Learning Models
The black-box nature of deep models hampers user trust and understanding.
- **Future directions:** Explainable AI (XAI), Vision-language models (VLMs), Counterfactual explanation generation.

### 📍 Anomaly Localization and Temporal Segmentation
Accurate spatiotemporal localization is often missing, especially in weakly supervised methods.
- **Challenges:** Frame-level granularity, vague reconstructions.
- **Future directions:** Spatiotemporal attention, Transformer-based architectures, Object-centric localization, Pixel-wise supervision.

### 📡 Integration with Multimodal Data (Audio, Sensors, Metadata)
Most current models are vision-only, ignoring useful modalities available in real-world settings.
- **Future directions:** Multimodal fusion, Cross-modal contrastive learning, Multisensor anomaly detection in smart environments.

### 🏷️ Data Annotation and Benchmark Limitations
Detailed annotations are costly and rare; most datasets use coarse labels.
- **Future directions:** Self-supervised learning techniques, Synthetic dataset creation.

### ⚖️ Ethical Considerations
Ethical issues include bias and surveillance-related privacy threats.
- **Solutions:** Federated learning, Differential privacy, Bias mitigation via augmentation and balanced sampling.

### 🌟 Emerging Trends and Future Vision
- Transformer models and attention-based systems are replacing older ConvLSTM/3D CNNs.
- Vision-Language Models (CLIP, BLIP, VERA) are enabling zero-shot anomaly detection.
- Federated learning is rising in importance for privacy-sensitive applications.
- Human-in-the-loop feedback systems for iterative anomaly refinement are gaining traction.
    """)
    
# Add footer to all pages
st.markdown("---")
st.markdown("⚠️ **Note:** This companion app does not execute or train any models. It is designed solely for exploration and comparative analysis based on existing literature and open-source tools.")