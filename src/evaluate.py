import torch
import numpy as np
import argparse
import time
import sys
from model import DensityMatrixTransformer
from data import ShadowDataset

def compute_fidelity(rho, sigma):
    """
    Computes Fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2
    Assumes rho, sigma are batches of density matrices or single matrices.
    """
   
    rho = rho.to(torch.complex64)
    sigma = sigma.to(torch.complex64)
    
    L_rho, V_rho = torch.linalg.eigh(rho)
   
    L_rho = torch.clamp(L_rho, min=0.0)
    sqrt_L_rho = torch.sqrt(L_rho)
    
    
    D_rho = torch.diag_embed(sqrt_L_rho.type_as(rho))
    sqrt_rho = V_rho @ D_rho @ V_rho.mH
    
    
    temp = sqrt_rho @ sigma @ sqrt_rho
    
    
    L_temp, V_temp = torch.linalg.eigh(temp)
    L_temp = torch.clamp(L_temp, min=0.0)
    
    D_temp = torch.diag_embed(torch.sqrt(L_temp).type_as(rho))
    sqrt_temp = V_temp @ D_temp @ V_temp.mH
    
   
    tr = torch.diagonal(sqrt_temp, dim1=-2, dim2=-1).sum(-1)
    
    
    fidelity = torch.real(tr)**2
    return fidelity

def compute_trace_distance(rho, sigma):
    """
    Trace Distance D(rho, sigma) = 0.5 * Tr(|rho - sigma|)
    |A| = sqrt(A_dag A)
    """
    diff = rho - sigma
  
    vals = torch.linalg.eigvalsh(diff)
    td = 0.5 * torch.sum(torch.abs(vals), dim=-1)
    return td

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
   
    model = DensityMatrixTransformer(n_qubits=args.n_qubits, 
                                     d_model=args.d_model, 
                                     nhead=args.nhead, 
                                     num_layers=args.num_layers).to(device)
    try:
        model.load_state_dict(torch.load("outputs/model.pt", map_location=device))
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first.")
        return

    model.eval()
    
   
    print(f"Generating {args.test_size} test samples...")
    test_data = ShadowDataset(size=args.test_size, n_qubits=args.n_qubits, n_shots=args.n_shots)
    
    
    fidelities = []
    trace_distances = []
    latencies = []
    
    print("Evaluating...")
    with torch.no_grad():
        for i in range(len(test_data)):
            noisy, clean = test_data[i]
            noisy = noisy.unsqueeze(0).to(device)
            clean_complex = torch.complex(clean[0], clean[1]).to(device)
            
            start = time.time()
            output_rho = model(noisy)
            end = time.time()
            
            latencies.append((end - start) * 1000) 
            
            output_rho = output_rho.squeeze(0) 
            output_complex = torch.complex(output_rho[0], output_rho[1])
            
            fid = compute_fidelity(output_complex, clean_complex)
            td = compute_trace_distance(output_complex, clean_complex)
            
            fidelities.append(fid.item())
            trace_distances.append(td.item())
            
    mean_fid = np.mean(fidelities)
    mean_td = np.mean(trace_distances)
    mean_lat = np.mean(latencies)
    
    print("\n=== Evaluation Results ===")
    print(f"Mean Fidelity: {mean_fid:.4f}")
    print(f"Mean Trace Distance: {mean_td:.4f}")
    print(f"Avg Inference Latency: {mean_lat:.2f} ms")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--n_shots", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    
    args = parser.parse_args()
    evaluate(args)
