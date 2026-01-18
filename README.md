# Assignment 2: Density Matrix Reconstruction
**Track 1: Classical Shadows (Transformer Model)**

**Author**: Macha Hareesh (Enrollment No: 24114056)

##  Objective
Develop a deep learning model to reconstruct a quantum density matrix $\rho$ from measurement data (Classical Shadows) while strictly enforcing physical constraints:
- **Hermitian**: $\rho = \rho^\dagger$
- **Positive Semi-Definite (PSD)**: $\rho \succeq 0$
- **Unit Trace**: $\text{Tr}(\rho) = 1$

##  Features
- **Transformer Architecture**: Captures global correlations in measurement data.
- **Cholesky Decomposition**: Output layer predicts $L$ such that $\rho = \frac{LL^\dagger}{\text{Tr}(LL^\dagger)}$, guaranteeing physical validity.
- **Single Qubit Support**: Optimized for $N=1$ reconstruction with high fidelity.

##  Repository Structure
```
/src
  ├── data.py       # Data generation (Shadows simulation)
  ├── model.py      # Transformer model + Cholesky output
  ├── train.py      # Training loop
  └── evaluate.py   # Metrics calculation (Fidelity, Trace Dist)
/outputs
  └── model.pt      # Trained model weights
/docs
  ├── MODEL_WORKING.md      # Methodological details
  ├── REPLICATION_GUIDE.md  # Step-by-step execution guide
  └── Assignment2_Report.pdf # Final Report
```

##  Setup & Usage

### 1. Install Dependencies
```bash
pip install torch numpy
```

### 2. Train the Model
To retrain the model (default: 1 qubit, 5000 samples, 50 epochs):
```bash
python src/train.py --n_qubits 1 --train_size 5000 --epochs 50
```

### 3. Evaluate Results
To verify performance on a test set:
```bash
python src/evaluate.py --n_qubits 1 --test_size 100
```

##  Results
The model achieves near-perfect reconstruction on the test set:

| Metric | Value | Ideal |
|--------|-------|-------|
| **Mean Fidelity** | **0.99** | 1.00 |
| **Mean Trace Distance** | **0.005** | 0.00 |
| **Inference Latency** | **3.24 ms** | - |

##  AI Attribution
This project uses AI tools for setup and debugging. See [AI_USAGE.md](AI_USAGE.md) for details.
