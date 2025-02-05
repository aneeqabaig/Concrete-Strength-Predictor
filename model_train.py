import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("concrete_data.csv")

# Drop missing values
df = df.dropna()

# Remove rows where Cement is zero (to avoid division errors)
df = df[df["Cement"] > 0]

# Compute Water-Cement Ratio
df["Water_Cement_Ratio"] = df["Water"] / df["Cement"]

# Select features & target
X = df[["Age", "Water_Cement_Ratio"]]
y = df["Strength"]

# Standardize features to prevent extreme values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a more powerful model (RandomForest instead of LinearRegression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model & scaler
joblib.dump(model, "cement_strength_model.pkl")
joblib.dump(scaler, "scaler.pkl")


