# Crop Yield Prediction using Machine Learning

## 1. Problem Statement

### Background
Agricultural productivity is a critical factor influencing food security and the economy of many nations. Farmers and policymakers often struggle to predict crop yield accurately due to varying environmental, soil, and climatic conditions. Traditional prediction methods rely heavily on expert judgment and historical averages, which are often inaccurate under changing climate patterns.

### Objectives
- To develop a data-driven model that predicts crop yield (tons per hectare) based on key environmental and agricultural features.
- To identify which factors (e.g., rainfall, temperature, pesticide use) most influence yield outcomes.
- To provide an easy-to-use **REST API** for yield prediction and decision support.

### Challenges
- Handling data inconsistencies and missing values across multiple countries and years.  
- Accounting for diverse climatic and soil conditions.  
- Selecting and tuning models that generalize well across different crops and regions.

### Expected Outcome
- A trained machine learning model capable of accurately predicting crop yield.  
- A user-friendly **FastAPI web service** for real-time predictions.  
- A reproducible, containerized ML pipeline deployable to the cloud.


---

## 2. Dataset Description

**Source:** [Crop Yield Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data)

### Dataset Highlights
- Contains crop production data for multiple countries and crops.
- Time range: Several decades (covering yearly statistics).
- Combines agricultural, climatic, and chemical use data.

### Features
| Feature | Description |
|----------|--------------|
| `` |  |
| `` |   |
| `` |  |
| `` |  |
| `` |  |
| `` | |
| `` |  |
| `` |  |

**Target Variable:**  
` ` — Yield per hectare (hectograms per hectare).

### Data Assets
- **Raw dataset:** 
- **Processed dataset:** Saved after cleaning and feature engineering.
- **Notebooks:**
  - 
  -
---

## 3. Technology Stack

| Layer | Tools / Libraries |
|--------|-------------------|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Modeling** | Scikit-learn, XGBoost |
| **Web Framework** | FastAPI / Flask |
| **Model Serialization** | Joblib |
| **Containerization** | Docker |
| **Cloud Deployment** | Hugging Face Spaces (Docker Space) |
| **Version Control** | Git, GitHub |

---

## 6. Folder Structure

crop-yield-predictor/
├── data/
│   └── 
├── models/
│   └── 
├── notebooks/
│   ├── 
│   └── 
├── src/
│   └── 
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore


---
## 5. Application Workflow

### Step 1: Data Preprocessing

### Step 2: Exploratory Data Analysis (EDA)

### Step 3: Model Training

### Step 4: Model Deployment (API Development)

### Step 5: Model Evaluation 

---

## 6. Instructions to Reproduce

### A. Run Locally (Not Codespaces)

**Step 1: Clone the Repository**
```bash
git clone https://github.com/<your-username>/crop-yield-prediction.git
cd crop-yield-prediction
```

**Step 2: Create and Activate Virtual Environment**
```bash
python -m venv venv
# Activate the environment
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run the FastAPI App**



API will be available at:

---

### B. Using Docker

**Step 1: Build the Docker Image**


**Step 2: Run the Docker Container**



Visit http://localhost:8000/docs for Swagger UI.

---

### C. Cloud Deployment

- Deploy the containerized app on Hugging Face Spaces.

- Push your project with Dockerfile to GitHub.

- On Hugging Face → Create a new Space → Choose Docker SDK.

- Link your GitHub repo or upload files manually.

- Build and run automatically on Hugging Face infrastructure.

App will be accessible at:

https://huggingface.co/spaces/<username>/crop-yield-prediction


--- 

## 8. Future Enhancements

- Integrate soil and humidity features for more robust predictions.

- Implement SHAP or LIME for feature explainability.

- Extend to time-series models (LSTM or Prophet) for future yield forecasting.

- Add an interactive dashboard frontend (React or Streamlit) consuming the API.

- Automate deployment with GitHub Actions + Docker CI/CD.