import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Streamlit title
st.title("Income Classification App 💼")
st.write("Predict whether income is >50K or <=50K based on user input")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult_sample.csv")  # Make sure this matches your file
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

df, label_encoders = load_data()

# Feature and target split
X = df.drop("income", axis=1)
y = df["income"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with K=1 since dataset is very small
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_scaled, y_train)

# Sidebar input for user
st.sidebar.header("Input User Data")
user_input = {}

for col in X.columns:
    if col in label_encoders:
        user_input[col] = st.sidebar.selectbox(f"{col}", label_encoders[col].classes_)
    else:
        user_input[col] = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode input
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Income"):
    prediction = model.predict(input_scaled)[0]
    result = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"Predicted Income: **{result}**")

# Accuracy display
y_pred = model.predict(X_test_scaled)
st.write(f"Model Accuracy: **{accuracy_score(y_test, y_pred) * 100:.2f}%**")

