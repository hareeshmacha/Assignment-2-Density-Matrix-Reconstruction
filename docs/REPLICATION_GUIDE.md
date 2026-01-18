# Replication Guide

## 1. Environment Setup
Ensure you have Python 3.8+ installed.

### Dependencies
Install the required packages:
```bash
pip install torch numpy
```
(Note: CUDA is recommended for training but not required. The code automatically detects GPU/CPU).

## 2. Repository Structure
- `src/`: Source code.
  - `data.py`: Data generation and dataset class.
  - `model.py`: Transformer model and Cholesky output layer.
  - `train.py`: Training script.
  - `evaluate.py`: Evaluation and metrics.
- `outputs/`: Stores trained model weights.
- `docs/`: Documentation.

## 3. Usage

### 3.1 Training the Model
To train the model, run the `train.py` script. You can specify parameters using command-line arguments.

**Command:**
```bash
python src/train.py --n_qubits 2 --train_size 1000 --epochs 10
```

**Arguments:**
- `--n_qubits`: Number of qubits (default: 2).
- `--train_size`: Number of training samples (default: 1000).
- `--epochs`: Number of training epochs (default: 10).
- `--batch_size`: Batch size (default: 32).
- `--lr`: Learning rate (default: 1e-3).

### 3.2 Evaluating the Model
After training, a model file `model.pt` will be saved in `outputs/`. Use `evaluate.py` to calculate metrics.

**Command:**
```bash
python src/evaluate.py --n_qubits 2 --test_size 100
```

**Output:**
The script will report:
- **Mean Fidelity**: Quantum fidelity between predicted and true states (Higher is better, max 1.0).
- **Mean Trace Distance**: Distance metric (Lower is better, min 0.0).
- **Inference Latency**: Average time per reconstruction (ms).

## 4. Troubleshooting
- **RuntimeError: expected scalar type...**: Ensure you are running `evaluate.py` with the updated code that handles complex type casting correctly.
- **Python not found**: Use `py -3` or ensure python is added to your PATH.
