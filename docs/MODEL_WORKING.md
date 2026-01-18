# Model Working: Density Matrix Reconstruction

## 1. Overview
This project implements a **Density Matrix Reconstruction** model using a **Transformer-based architecture**. The goal is to predict the true quantum state $\rho$ from noisy measurement data (simulated Classical Shadows).

## 2. Mathematical Logic
### 2.1 Problem Formulation
Given a set of measurements (e.g., Pauli expectation values or shadow snapshots), we aim to find $\hat{\rho}$ that minimizes the distance to the true state.
In our implementation, we treat the input as a "noisy" estimate of the density matrix (derived from averaging shadow snapshots). The model acts as a denoising autoencoder:
$$ \mathcal{F}(\rho_{noisy}) \rightarrow \rho_{clean} $$

### 2.2 Physical Constraints
A valid density matrix must satisfy:
1.  **Hermitian**: $\rho = \rho^\dagger$
2.  **Positive Semi-Definite (PSD)**: $v^\dagger \rho v \geq 0 \quad \forall v$
3.  **Unit Trace**: $\text{Tr}(\rho) = 1$

To enforce these constraints strictly, we use the **Cholesky Decomposition** method in the output layer. Instead of predicting $\rho$ directly, the model predicts a lower triangular matrix $L$ (with complex entries).
We construct $\rho$ as:
$$ \rho_{unnormalized} = L L^\dagger $$
This construction guarantees that $\rho_{unnormalized}$ is Hermitian and PSD by definition.
Finally, we normalize the trace:
$$ \rho = \frac{\rho_{unnormalized}}{\text{Tr}(\rho_{unnormalized})} $$

## 3. Architecture
### 3.1 Input Layer
The input is a complex matrix of size $(N, N)$ where $N=2^n$. We represent this as a tensor calculation with real and imaginary channels:
- Input Shape: `(Batch, 2, N, N)`

### 3.2 Transformer Encoder
We flatten the density matrix rows into a sequence of vectors.
- Each row of the $N \times N$ matrix is treated as a token.
- Sequence Length: $N$
- Token Dimension: $2 \times N$ (Real + Imag parts)
- The transformer captures correlations between different basis states.

### 3.3 Cholesky Output Head
The transformer output is projected to fit the parameters of $L$.
- $L$ has $N(N+1)/2$ complex parameters (or $N^2$ if we don't strictly mask, but we mask the upper triangle).
- We predict $L_{real}$ and $L_{imag}$.
- We compute $\rho = (L_{real} + iL_{imag})(L_{real}^T - iL_{imag}^T)$ using real-valued matrix multiplications to ensure stability and autograd compatibility.

## 4. Training
- **Loss Function**: Mean Squared Error (MSE) on the real and imaginary components of $\rho$.
$$ \mathcal{L} = ||\rho_{pred} - \rho_{true}||_F^2 $$
- **Optimizer**: Adam
