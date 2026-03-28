# 🔩 ML-based Material Selection for Industrial Components

> Unsupervised machine learning to cluster engineering materials by mechanical properties — helping Siemens Energy engineers identify the right material for turbines, engines, and heavy machinery faster.

---

## 📌 Overview

This project was developed as part of **UCS321 – AI for Engineers**, with a theoretical use case inspired by **Siemens Energy**. Given a dataset of materials with mechanical properties (tensile strength, yield strength, hardness, density, thermal conductivity, elasticity), we apply **K-Means clustering** to group materials with similar characteristics.

The goal is to reduce material selection time for engineers by surfacing pre-clustered material groups mapped to real-world component types — lightweight parts, standard machinery, and engine/turbine components.

---

## 🗂️ Dataset

- **Source:** [Materials Dataset – Kaggle](https://www.kaggle.com/datasets/purushottamnawale/materials)
- **Features used:** Tensile Strength, Yield Strength, Hardness, Density, Thermal Conductivity, Elasticity
- **Preprocessing:** Missing values filled with column medians; features standardized using `StandardScaler`

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML (GPU) | NVIDIA cuML (KMeans, PCA, t-SNE, StandardScaler) |
| ML (CPU fallback) | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Google Colab |
| Model Saving | Joblib |

> ⚡ The notebook auto-detects GPU availability and uses **cuML** for accelerated training, falling back to **scikit-learn** on CPU automatically.

---

## 🚀 How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Upload `AISiemens_Code.ipynb` to Google Colab
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/purushottamnawale/materials) and upload `Data.csv` when prompted
3. Run all cells top to bottom

### Option 2 — Run Locally

```bash
# Clone the repo
git clone https://github.com/Abhishek2405079/ML-based-Material-Selection.git
cd ML-based-Material-Selection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook AISiemens_Code.ipynb
```

---

## 🧠 Methodology

```
Raw Material Data (CSV)
        │
        ▼
Data Cleaning & Median Imputation
        │
        ▼
Feature Scaling (StandardScaler)
        │
        ▼
Elbow Method → Optimal K = 3
        │
        ▼
K-Means Clustering (3 clusters)
        │
        ▼
Dimensionality Reduction (PCA 2D/3D, t-SNE)
        │
        ▼
Cluster Labeling → Component Mapping
        │
        ▼
Material Recommendation Function
```

---

## 📊 Results

The Elbow Method identified **K = 3** as the optimal number of clusters. The clusters were mapped to the following industrial component categories:

| Cluster | Component Type | Typical Materials |
|---|---|---|
| 0 | ⚙️ Lightweight Parts | Aluminum alloys, composites |
| 1 | 🔧 Standard Machinery | Steel alloys, cast iron |
| 2 | 🔥 Engine / Turbine | Titanium, Nickel superalloys |

A built-in recommendation function lets you query by component type using short codes:

```python
recommend_material_by_component('Et')   # Engine / Turbine
recommend_material_by_component('Lp')   # Lightweight Parts
recommend_material_by_component('Sm')   # Standard Machinery
```

Outputs are also saved to `Cluster_Summary.csv` for further analysis.

---

## 📁 Project Structure

```
ML-based-Material-Selection/
├── ML-based Material Selection.py   ← Main notebook
├── requirements.txt                 ← Python dependencies
├── .gitignore
├── README.md
└── Cluster_Summary.csv              ← Generated output (after running)
```

> **Note:** `Data.csv`, `kmeans_model.pkl`, and `scaler.pkl` are excluded from the repo via `.gitignore`. Download the dataset from Kaggle and run the notebook to regenerate them.

---

## 👥 Team

Developed as a group project by 5 students from the **UCS321 – AI for Engineers** course.

---

## 📄 License

This project is for academic purposes under the UCS321 course. Dataset credit: [Purushottam Nawale on Kaggle](https://www.kaggle.com/datasets/purushottamnawale/materials).
