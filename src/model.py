import torch
import torch.nn as nn
import torch.nn.functional as F

class CholeskyOutputLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
       
        batch_size = x.shape[0]
        x = x.view(batch_size, self.dim, self.dim, 2)
        
        
        A = x[..., 0] 
        B = x[..., 1]
        
        tril_mask = torch.tril(torch.ones(self.dim, self.dim, device=x.device))
        A = A * tril_mask
        B = B * tril_mask
        
       
        
        rho_real = torch.matmul(A, A.transpose(-1, -2)) + torch.matmul(B, B.transpose(-1, -2))
        rho_imag = torch.matmul(B, A.transpose(-1, -2)) - torch.matmul(A, B.transpose(-1, -2))
        
       
        
        trace = torch.diagonal(rho_real, dim1=-2, dim2=-1).sum(-1)
        trace = trace.clamp(min=1e-6)
        
        rho_real = rho_real / trace.view(-1, 1, 1).expand_as(rho_real)
        rho_imag = rho_imag / trace.view(-1, 1, 1).expand_as(rho_imag)
        
        return torch.stack([rho_real, rho_imag], dim=1)

class DensityMatrixTransformer(nn.Module):
    def __init__(self, n_qubits=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.dim = 2**n_qubits
        self.input_dim = 2 * self.dim 
        
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_dim = 2 * self.dim * self.dim
        self.fc_out = nn.Linear(d_model * self.dim, self.output_dim)
        
        self.cholesky_layer = CholeskyOutputLayer(self.dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.dim, -1) 
        
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        x = self.fc_out(x)
        
        rho = self.cholesky_layer(x)
        return rho
