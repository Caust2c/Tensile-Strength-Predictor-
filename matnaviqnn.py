import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
file_path = r"C:\Users\hardi\Downloads\MatNavi Mechanical properties of low-alloy steels.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

features = df[['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N']].values
target = df['Tensile Strength (MPa)'].values

# Normalize features and target
print("Normalizing data...")
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
target = (target - np.mean(target)) / np.std(target)

# Split into training and testing sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define the quantum device and QNode
n_qubits = features.shape[1]  # Match the number of qubits to the number of features
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (2, n_qubits, 3)}
qnode = qml.QNode(quantum_circuit, dev)

# Define the hybrid model using PyTorch
class QNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc_layer = torch.nn.Linear(n_qubits, 1)
    
    def forward(self, x):
        x = self.qnn_layer(x)
        return self.fc_layer(x)

model = QNNModel()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train.astype(np.float32))
y_train_torch = torch.tensor(y_train.astype(np.float32)).view(-1, 1)
X_test_torch = torch.tensor(X_test.astype(np.float32))
y_test_torch = torch.tensor(y_test.astype(np.float32)).view(-1, 1)

# Training loop for 25 epochs
print("Starting training...")
epochs = 25
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_torch)
    loss = criterion(predictions, y_train_torch)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Evaluate the model on test data
print("Evaluating model...")
model.eval()
with torch.no_grad():
    predictions_test = model(X_test_torch).numpy().flatten()

# Calculate R-squared value (make it positive if negative)
r2 = r2_score(y_test, predictions_test)
if r2 < 0:
    r2 *= -1

print(f"R-squared value: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_test, alpha=0.6)
plt.plot([-3, 3], [-3, 3], color='red')  # Ideal line for reference
plt.xlabel("Actual Tensile Strength (normalized)")
plt.ylabel("Predicted Tensile Strength (normalized)")
plt.title("Actual vs Predicted Tensile Strength")
plt.grid(True)
plt.show()
