import numpy as np
import torch
from torch.utils.data import Dataset

def random_density_matrix(dim):
    """
    Generates a random valid density matrix of dimension `dim` using the 
    Ginibre ensemble (Hilbert-Schmidt measure).
    """
    
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
   
    rho = np.dot(A, A.conj().T)
   
    rho /= np.trace(rho)
    return rho.astype(np.complex64)

def get_pauli_matrices():
    I = np.eye(2, dtype=np.complex64)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    return I, X, Y, Z

I, X, Y, Z = get_pauli_matrices()
PAULIS = {'X': X, 'Y': Y, 'Z': Z}


def measure_in_basis(rho, basis_index):
    """
    Simulates a measurement of single qubit rho in X, Y, or Z basis.
    basis_index: 0=X, 1=Y, 2=Z
    Returns: outcome (+1 or -1), and the post-measurement state vector (as density matrix for shadow calc).
    """
    
    pass 

def fidelity(rho1, rho2):
    
    sqrt_rho1 = sqrtm(rho1)
    temp = np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)
    return np.real(np.trace(sqrtm(temp)))**2

def sqrtm(acc):
    
    w, v = np.linalg.eig(acc)
    
    sqrt_w = np.sqrt(w)
    return np.dot(v * sqrt_w, v.conj().T)

class ShadowDataset(Dataset):
    def __init__(self, size, n_qubits, n_shots):
        self.size = size
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.dim = 2**n_qubits
        
        self.data = []
        print(f"Generating {size} samples for n_qubits={n_qubits}...")
        for _ in range(size):
            rho = random_density_matrix(self.dim)
            shadow_rho = self.simulate_shadow_reconstruction(rho)
            self.data.append((shadow_rho, rho))
            
    def simulate_shadow_reconstruction(self, rho):
        
        running_sum = np.zeros((self.dim, self.dim), dtype=np.complex64)
        
       
        
        noise_level = 1.0 / np.sqrt(self.n_shots)
       
        H = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        H = (H + H.conj().T) / 2
       
        H = H - np.trace(H) * np.eye(self.dim) / self.dim
        
        rho_noisy = rho + noise_level * H
        return rho_noisy.astype(np.complex64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]
      
        
        noisy_tensor = torch.from_numpy(np.stack([noisy.real, noisy.imag], axis=0)).float()
        clean_tensor = torch.from_numpy(np.stack([clean.real, clean.imag], axis=0)).float()
        
        return noisy_tensor, clean_tensor
