# Income Classification using K-Nearest Neighbors (KNN)

This project predicts whether an individual earns more than $50,000 per year based on census data using the **K-Nearest Neighbors (KNN)** classification algorithm. The app is built using **Streamlit** for interactive UI.

---

## 🚀 Project Overview

The goal of this project is to:
- Predict income class (`<=50K` or `>50K`)
- Use user-input features like age, education, workclass, etc.
- Build a machine learning pipeline using **KNN**
- Deploy a simple user interface using **Streamlit**

---

## 📁 Dataset

The dataset used is a subset of the [Adult Income Dataset (UCI Census Data)](https://archive.ics.uci.edu/ml/datasets/adult), saved locally as:

- `adult_sample.csv`

---

## 🧠 Model Used

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Preprocessing**:
  - Categorical encoding: `LabelEncoder`
  - Feature scaling: `StandardScaler`
- **k = 1**

---

## 🛠️ How It Works

1. Load and clean the dataset (`adult_sample.csv`)
2. Encode categorical features using `LabelEncoder`
3. Scale numeric features using `StandardScaler`
4. Train a KNN classifier with `k=1`
5. Create Streamlit UI for user inputs
6. Predict income class using the trained model
7. Display prediction result and model accuracy

---

## 🖥️ Run the App

To run the Streamlit app:

```bash
pip install streamlit pandas scikit-learn
streamlit run streamlit_app.py
