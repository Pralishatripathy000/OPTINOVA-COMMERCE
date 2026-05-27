<div align="center">

<!-- Replace with your launch page screenshot -->
![Optinova Commerce Banner](images/launch_page.png)

<br>

# Optinova Commerce
### AI-Powered Ecommerce Conversion & Cart Abandonment Intelligence

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://optinova-commerce-kinyycrkxse3hw32xqicmq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

**Predict. Prevent. Convert.**

*An end-to-end AI-powered ecommerce intelligence web application that predicts purchase intent and scores cart abandonment risk in real time — built from raw data to a deployed, interactive dashboard.*

<br>

[🚀 Live Demo](https://optinova-commerce-kinyycrkxse3hw32xqicmq.streamlit.app/) · [📊 Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) · [🐛 Report Bug](../../issues)

</div>

---

<br>

## 📌 Overview

Optinova Commerce is an AI-powered ecommerce analytics platform that analyzes visitor session behavior and predicts whether a customer is likely to complete a purchase or abandon their shopping journey.

The system combines **machine learning**, **behavioral analytics**, and **real-time risk scoring** to surface actionable conversion intelligence — built for innovative brands and platform-agnostic by design.

> **Example:** If a visitor explores multiple product pages but shows high bounce and exit behavior, the model detects elevated abandonment risk and surfaces intelligent behavioral insights — in real time.

<br>

<!-- Replace with your dashboard overview screenshot -->
![Dashboard Overview](images/dashboard_overview.png)

<br>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **AI Purchase Prediction** | Predicts purchase likelihood per session with 92.7% model accuracy |
| ⚠️ **Abandonment Risk Scoring** | Generates a 0–100 abandonment risk score per live session |
| 📊 **Behavioral Analytics** | Computes engagement score, buyer intent, session health, revenue potential |
| 📈 **Interactive Visualizations** | Gauge charts, radar charts, conversion funnels, feature importance bars |
| 📄 **AI Session Report** | One-click downloadable PDF report per session |
| 🌐 **Platform Agnostic** | Integrable with any ecommerce, SaaS, EdTech, or digital platform |

<br>

---

## 📸 Screenshots

<br>

### Hero Page
<!-- Replace with your hero / launch page screenshot -->
![Hero Page](images/hero_page.png)

<br>

### Live Prediction Dashboard
<!-- Replace with your prediction page screenshot -->
![Prediction Dashboard](images/prediction_dashboard.png)

<br>

### Model Insights — Feature Importance
<!-- Replace with your feature importance screenshot -->
![Feature Importance](images/feature_importance.png)

<br>

### Risk Breakdown
<!-- Replace with your risk breakdown screenshot -->
![Risk Breakdown](images/risk_breakdown.png)

<br>

### About Page
<!-- Replace with your about page screenshot -->
![About Page](images/about_page.png)

<br>

---

## 🏗️ Architecture

<br>

<!-- Replace with your architecture diagram -->
![Architecture Diagram](images/architecture_diagram.png)

<br>

```
Raw Dataset
    │
    ▼
Data Cleaning & EDA
(Pandas, NumPy, Matplotlib, Seaborn)
    │
    ▼
Feature Engineering
(EngagementScore, ExitIntentScore, One-Hot Encoding)
    │
    ▼
Model Training
(XGBoost Classifier — 200 estimators, max_depth 6, lr 0.1)
    │
    ▼
Model Serialization
(joblib → .pkl files: model, scaler, features)
    │
    ▼
Streamlit Web Application
(Multi-page dashboard, Plotly charts, HTML5/CSS3 panels)
    │
    ▼
Deployed on Streamlit Cloud
```

<br>

---

## 🔄 ML Workflow

<br>

<!-- Replace with your ML workflow diagram -->
![ML Workflow](images/ml_workflow.png)

<br>

### Step by Step

**1. Dataset**
UCI Online Shoppers Purchasing Intention Dataset — 12,330 sessions, 18 features, binary classification target (`Revenue`).

**2. EDA**
Explored Revenue distribution, Bounce Rate vs Revenue, Exit Rate vs Revenue, Visitor Type patterns, Month-wise conversion trends, and Traffic Type analysis.

**3. Feature Engineering**
- `EngagementScore` = `(ProductRelated × PageValues) / (BounceRates + 1)`
- `ExitIntentScore` = `BounceRates + ExitRates`
- One-hot encoding on `Month`, `VisitorType`, `Weekend`

**4. Model Training**
XGBoost Classifier trained on 80/20 train-test split with StandardScaler normalization.

**5. Abandonment Risk Score**
`AbandonmentRiskScore = (1 − PurchaseProbability) × 100`

**6. Dashboard**
Multi-page Streamlit app with live session input, real-time prediction, and downloadable PDF report.

<br>

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 92.7% |
| **Model** | XGBoost Classifier |
| **Estimators** | 200 |
| **Max Depth** | 6 |
| **Learning Rate** | 0.1 |
| **Train/Test Split** | 80% / 20% |
| **Scaling** | StandardScaler |

<br>

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Frontend** | Streamlit, HTML5, CSS3 |
| **Fonts** | Google Fonts API (Manrope) |
| **Report Generation** | ReportLab |
| **Model Serialization** | Joblib |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git, GitHub |

<br>

---

## 📁 Project Structure

```
optinova-commerce/
│
├── optinovacomm.py               # Main Streamlit application
├── OPTINOVA_COMM_INITIAL.py      # Original Jupyter notebook (training pipeline)
│
├── optinova_xgb_model.pkl        # Trained XGBoost model
├── optinova_scaler.pkl           # Fitted StandardScaler
├── optinova_features.pkl         # Feature column names
├── optinova_logo.png             # Brand logo
│
├── online_shoppers_intention.csv # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

<br>

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/optinova-commerce.git
cd optinova-commerce
```

**2. Create and activate a conda environment**
```bash
conda create -n optinova_env python=3.10
conda activate optinova_env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the app**
```bash
streamlit run optinovacomm.py
```

**5. Open in browser**
```
http://localhost:8501
```

<br>

---

## 📦 Requirements

```
streamlit
pandas
numpy
xgboost
scikit-learn
plotly
streamlit-option-menu
reportlab
joblib
matplotlib
seaborn
```

<br>

---

## 🌍 Real World Applications

Optinova Commerce is platform-agnostic — the behavioral prediction engine can be integrated with any digital platform depending on the data pipeline and tech stack.

| Industry | Use Case |
|---|---|
| 🛍️ **Ecommerce Stores** | Reduce cart abandonment, increase revenue |
| ✈️ **Travel & Hospitality** | Predict booking drop-off, re-engage users |
| 🏦 **Finance & Banking** | Detect application abandonment |
| 🎓 **EdTech Platforms** | Identify high-intent students |
| 👑 **SaaS & Subscriptions** | Reduce churn, improve renewals |
| 🎟️ **Events & Ticketing** | Predict intent before checkout abandonment |

<br>

---

## 🗺️ Roadmap

- [x] XGBoost model training and evaluation
- [x] Abandonment risk scoring logic
- [x] Multi-page Streamlit dashboard
- [x] PDF AI session report export
- [x] Deployment on Streamlit Cloud
- [ ] SHAP explainability layer
- [ ] Batch prediction (CSV upload)
- [ ] Backend API layer (FastAPI)
- [ ] Database integration
- [ ] Full stack deployment

<br>

---

## 👩‍💻 Author

**Pralisha Tripathy**
5-Year Integrated MTech | VIT

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)

<br>

---

<div align="center">

**Made with 🧠 and way too much CSS debugging.**

*Built for innovative brands — Amazon · Flipkart · Myntra · Meesho · Nykaa · Tata CLiQ · Zepto*

⭐ Star this repo if you found it useful

</div>
