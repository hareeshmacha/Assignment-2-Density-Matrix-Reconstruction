import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import torch.nn.functional as F

from data import ShadowDataset
from model import DensityMatrixTransformer

def complex_mse_loss(output, target):
   
    loss = F.mse_loss(output, target)
    return loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
   
    print("Generating training data...")
    train_dataset = ShadowDataset(size=args.train_size, n_qubits=args.n_qubits, n_shots=args.n_shots)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
   
    model = DensityMatrixTransformer(n_qubits=args.n_qubits, 
                                     d_model=args.d_model, 
                                     nhead=args.nhead, 
                                     num_layers=args.num_layers).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (noisy, clean) in enumerate(train_loader):
           
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
           
            output_rho = model(noisy) 
            
            
            loss = complex_mse_loss(output_rho, clean)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f}s")
    
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--n_shots", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    
    args = parser.parse_args()
    train(args)
