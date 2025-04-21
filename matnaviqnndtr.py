import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
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

X = df[selected_features].values
y = df[target_column].values.reshape(-1, 1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

n_components = 4
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_pca)
X_val_scaled = scaler_x.transform(X_val_pca)
X_test_scaled = scaler_x.transform(X_test_pca)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

n_qubits = n_components
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_tree_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qnode = qml.QNode(quantum_tree_circuit, dev)

class QuantumTreeRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.post_process = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        x = self.qlayer(x)
        return self.post_process(x.reshape(-1, 1))

model = QuantumTreeRegressor()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
epochs = 300
loss_history = []

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test_t).numpy()
    test_pred = scaler_y.inverse_transform(test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

r2 = r2_score(y_test_orig, test_pred)
mse = mean_squared_error(y_test_orig, test_pred)
rmse = pow(mse, 0.5)
print(f"RMSE:  {rmse:.4f}")
print(f"\nQuantum Decision Tree Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

plt.figure(figsize=(10, 7))
plt.scatter(y_test_orig, test_pred, color="red", alpha=0.7)
plt.plot([y_test_orig.min(), y_test_orig.max()], 
         [y_test_orig.min(), y_test_orig.max()], 
         linestyle="--", color="blue", label="Ideal Prediction")
plt.title("Quantum Decision Tree: Predicted vs Actual", fontsize=14)
plt.xlabel("Actual Tensile Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Tensile Strength (MPa)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.tight_layout()
plt.show()
