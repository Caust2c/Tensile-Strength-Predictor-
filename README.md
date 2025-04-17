# Tensile-Strength-Predictor-
This repository contains an academic project that predicts the **tensile strength of low-alloy steels** using a combination of classical and quantum machine learning models.

It utilizes a dataset from the **MatNavi Mechanical Properties Database** and includes both **traditional regressors** and **quantum neural networks (QNNs)** built with PennyLane and PyTorch.

## 📁 Project Structure

| File | Model | Description |
|------|-------|-------------|
| `matnavicnn.py` | MLPRegressor | Classical neural network using ReLU and Adam optimizer |
| `matnavicnndtr.py` | DecisionTreeRegressor | Simple decision tree for regression |
| `matnavicnnrf.py` | RandomForestRegressor | Ensemble model using 100 decision trees |
| `matnavicnnsvr.py` | Support Vector Regression (SVR) | SVR with RBF kernel |
| `matnavicnnadaboost.py` | AdaBoost Regressor | Ensemble of boosted decision trees |
| `matnaviqnn.py` | Quantum Neural Network | Full-feature QNN using PennyLane and PyTorch |
| `matnaviqnndtr.py` | Quantum Decision Tree | decision tree regressor inspired QNN |

Each model includes:
- Feature preprocessing
- Training/validation/test split
- Model training
- R² and MSE evaluation
- Actual vs. predicted scatter plot

## Technologies & Libraries used

- `scikit-learn` – classical ML models (MLP, SVR, DecisionTree, AdaBoost, RandomForest)
- `pennylane` – quantum machine learning circuits
- `torch` –  deep learning framework
- `pandas`, `numpy` – data processing
- `matplotlib` – visualizing results

## 📊 Dataset Availability

The dataset used in this project is from the **MatNavi Mechanical Properties Database** (by NIMS, Japan). It contains chemical compositions and physical properties of low-alloy steels.

- 📂 GitHub Mirror: [MatNavi Mechanical Properties of Low-Alloy Steels (CSV)](https://github.com/ArtemRamus/PORTFOLIO-Mechanical-properties-of-low-alloy-steels/blob/main/MatNavi%20Mechanical%20properties%20of%20low-alloy%20steels.csv)
- **Format**: CSV

## 🛠 Installation

To run this project, you need to install the required dependencies. You can do this by creating a virtual environment and installing the packages from the `requirements.txt` file.

### Step-by-step:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Tensile-Strength-Predictor.git
   cd Tensile-Strength-Predictor
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Models
To run any of the models, simply run the corresponding Python file:
```bash
python matnavicnn.py
