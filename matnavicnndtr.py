import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the data (adjust file path as needed)
file_path = r"C:\filepath"
df = pd.read_csv(file_path)

# Clean column names (remove any leading/trailing spaces)
df.columns = df.columns.str.strip()

# Select features and target column
selected_features = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V",
    "Al", "N", "Ceq", "Nb + Ta", "Temperature (°C)",
    "0.2% Proof Stress (MPa)", "Elongation (%)", "Reduction in Area (%)"
]
target_column = "Tensile Strength (MPa)"

# Split into features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# Split dataset: 70% train, 30% temporary set (to later split to validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Further split the temporary set into 50/50 for validation (15%) and testing (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features (Scaling is not strictly necessary for tree-based models, but we use it for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Initialize and train a Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Predict on the test set
test_pred = decision_tree.predict(X_test_scaled)

# Calculate evaluation metrics: R² and Mean Squared Error
r2 = r2_score(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Plot actual vs predicted tensile strength
plt.figure(figsize=(10, 7))
plt.scatter(y_test, test_pred, color="red", alpha=0.7, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle="--", color="blue", label="Ideal Prediction")

plt.title("Predicted vs Actual Tensile Strength (Decision Tree)", fontsize=14)
plt.xlabel("Actual Tensile Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Strength (MPa)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Annotate the plot with the R² value
plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.tight_layout()

# Save and display the plot
plt.savefig('tensile_strength_prediction_decision_tree.png')
plt.show()
