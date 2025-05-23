import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import matplotlib.pyplot as plt

file_path = r"C:\filepath"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

selected_features = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V",
    "Al", "N", "Ceq", "Nb + Ta", "Temperature (°C)",
    "0.2% Proof Stress (MPa)", "Elongation (%)", "Reduction in Area (%)"
]
target_column = "Tensile Strength (MPa)"

X = df[selected_features]
y = df[target_column]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

param_dist = {
    'hidden_layer_sizes': [(64, 32, 16), (100,), (128, 64), (50, 50, 50)],
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs'],
    'max_iter': [300, 500, 1000]
}

mlp = MLPRegressor(random_state=42, early_stopping=True)

search = RandomizedSearchCV(
    mlp,
    param_distributions=param_dist,
    n_iter=20,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1,
    cv=3,
    verbose=0,
    random_state=42
)

search.fit(X_train_scaled, y_train)

best_model = search.best_estimator_
test_pred = best_model.predict(X_test_scaled)

r2 = r2_score(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 7))
plt.scatter(y_test, test_pred, color="red", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color="blue", label="Ideal Prediction")
plt.title("Predicted vs Actual Tensile Strength", fontsize=14)
plt.xlabel("Actual Tensile Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Strength (MPa)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.savefig('tensile_strength_prediction.png')
plt.show()
