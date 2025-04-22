import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import randint

file_path = r"C:\\filepath"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df["Cr_Ni"] = df["Cr"] / (df["Ni"] + 1e-6)
df["Ceq_Temp"] = df["Ceq"] / (df["Temperature (°C)"] + 1e-6)
df["Mn_Si"] = df["Mn"] * df["Si"]
df["Cr_Mo"] = df["Cr"] * df["Mo"]
df["C_squared"] = df["C"] ** 2
df["log_Mn"] = np.log1p(df["Mn"])
df["sqrt_Nb_Ta"] = np.sqrt(df["Nb + Ta"])
alloy_elements = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "Al", "N"]
df["Alloy_sum"] = df[alloy_elements].sum(axis=1)
df["Alloy_std"] = df[alloy_elements].std(axis=1)
df["Alloy_max"] = df[alloy_elements].max(axis=1)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Alloy_cluster"] = kmeans.fit_predict(df[alloy_elements])

selected_features = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "Al", "N", "Ceq", "Nb + Ta", "Temperature (°C)",
    "0.2% Proof Stress (MPa)", "Elongation (%)", "Reduction in Area (%)", "Cr_Ni", "Ceq_Temp", "Mn_Si", "Cr_Mo",
    "C_squared", "log_Mn", "sqrt_Nb_Ta", "Alloy_sum", "Alloy_std", "Alloy_max", "Alloy_cluster"
]
target_column = "Tensile Strength (MPa)"
X = df[selected_features]

X.loc[:, "Ceq_squared"] = X["Ceq"] ** 2
X.loc[:, "Ni_Cr_ratio"] = X["Ni"] / (X["Cr"] + 1e-6)
X.loc[:, "Mo_plus_V"] = X["Mo"] + X["V"]
X.loc[:, "Nb_Ta_times_N"] = X["Nb + Ta"] * X["N"]
X.loc[:, "Temp_log"] = np.log(X["Temperature (°C)"] + 1)
X.loc[:, "Elongation_to_Area"] = X["Elongation (%)"] / (X["Reduction in Area (%)"] + 1e-6)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = df[target_column]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': [None, 'sqrt', 'log2']
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rand_search = RandomizedSearchCV(rf_base, param_distributions=param_dist, n_iter=50, cv=5, verbose=0, scoring='r2', random_state=42, n_jobs=-1)
rand_search.fit(X_train_scaled, y_train)

rf_model = rand_search.best_estimator_
print(f"Best Hyperparameters for Random Forest: {rand_search.best_params_}")

test_pred = rf_model.predict(X_test_scaled)
r2 = r2_score(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, test_pred) * 100

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(10, 7))
plt.scatter(y_test, test_pred, color="red", alpha=0.7, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color="blue", label="Ideal Prediction")
plt.title("Predicted vs Actual Tensile Strength (Random Forest)", fontsize=14)
plt.xlabel("Actual Tensile Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Strength (MPa)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.tight_layout()
plt.savefig('tensile_strength_prediction_RF.png')
plt.show()
