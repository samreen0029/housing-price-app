import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡")

# Title
st.title("🏡 Housing Price Prediction using Ridge Regression")

st.write("This app predicts house prices based on input features.")

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# Features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Sidebar inputs
st.sidebar.header("Enter House Details")

inputs = {
    "MedInc": st.sidebar.slider("Median Income", 0.0, 15.0, 3.0),
    "HouseAge": st.sidebar.slider("House Age", 1.0, 50.0, 20.0),
    "AveRooms": st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0),
    "AveBedrms": st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0),
    "Population": st.sidebar.slider("Population", 100.0, 5000.0, 1000.0),
    "AveOccup": st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0),
    "Latitude": st.sidebar.slider("Latitude", 32.0, 42.0, 36.0),
    "Longitude": st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)
}

# Convert input
input_df = pd.DataFrame([inputs])

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = ridge.predict(input_scaled)

# Output
st.subheader("💰 Predicted House Price")
st.success(f"Estimated Value: ${prediction[0]*100000:.2f}")

# Model performance
if st.checkbox("Show Model Performance"):
    y_pred = ridge.predict(X_test_scaled)
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

# Feature importance
if st.checkbox("Show Feature Importance"):
    coeff_df = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])
    st.dataframe(coeff_df)
