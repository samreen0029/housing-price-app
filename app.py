import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡")

# Title
st.title("🏡 Housing Price Prediction (Ridge Regression)")
st.write("Predict house prices using demographic and geographic features")

# Load dataset
df = pd.read_csv("housing.csv")

# Handle missing values
df = df.dropna()

# Rename columns (Kaggle → ML format)
df = df.rename(columns={
    "median_income": "MedInc",
    "housing_median_age": "HouseAge",
    "total_rooms": "AveRooms",
    "total_bedrooms": "AveBedrms",
    "population": "Population",
    "households": "AveOccup",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "median_house_value": "MedHouseVal"
})

# Drop non-numeric column
if "ocean_proximity" in df.columns:
    df = df.drop(columns=["ocean_proximity"])

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# Missing values check
if st.checkbox("Check Missing Values"):
    st.write(df.isnull().sum())

# Statistics
if st.checkbox("Show Statistics"):
    st.write(df.describe())

# Heatmap
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Features and target
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

X = df[features]
y = df["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Sidebar inputs
st.sidebar.header("Enter House Details")

input_data = {
    "MedInc": st.sidebar.slider("Median Income", 0.0, 15.0, 3.0),
    "HouseAge": st.sidebar.slider("House Age", 1.0, 50.0, 20.0),
    "AveRooms": st.sidebar.slider("Average Rooms", 1.0, 10000.0, 5000.0),
    "AveBedrms": st.sidebar.slider("Average Bedrooms", 1.0, 5000.0, 1000.0),
    "Population": st.sidebar.slider("Population", 100.0, 50000.0, 1000.0),
    "AveOccup": st.sidebar.slider("Households", 1.0, 10000.0, 500.0),
    "Latitude": st.sidebar.slider("Latitude", 32.0, 42.0, 36.0),
    "Longitude": st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)
}

input_df = pd.DataFrame([input_data])

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)

# Output
st.subheader("💰 Predicted House Price")
st.success(f"Estimated Value: ${prediction[0]:,.2f}")

# Model performance
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test_scaled)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

# Feature importance
if st.checkbox("Show Feature Importance"):
    coeff_df = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
    st.dataframe(coeff_df)

# Actual vs Predicted plot
if st.checkbox("Show Actual vs Predicted Plot"):
    y_pred = model.predict(X_test_scaled)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
