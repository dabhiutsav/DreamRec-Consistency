import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConsistencyModel(nn.Module):
    def __init__(self, hidden_size, sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)

        c_skip, c_out, c_in = self.get_scalings(t)
        c_skip = c_skip.view(-1, 1)
        c_out = c_out.view(-1, 1)
        c_in = c_in.view(-1, 1)

        x_scaled = c_in * x

        x_input = torch.cat([x_scaled, t_emb], dim=1)

        model_output = self.model(x_input)

        output = c_skip * x + c_out * model_output
        
        return output

class ConsistencyTraining:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
    
    def get_timesteps(self, num_steps):
        steps = torch.arange(num_steps, dtype=torch.float32)
        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        timesteps = (t_max_rho + steps / (num_steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        return timesteps
    
    def consistency_loss(self, model, x_start, h, num_scales=50):
        device = x_start.device
        batch_size = x_start.shape[0]
        
        indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
        
        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        
        t = (t_max_rho + indices / (num_scales - 1) * (t_min_rho - t_max_rho)) ** self.rho
        t2 = (t_max_rho + (indices + 1) / (num_scales - 1) * (t_min_rho - t_max_rho)) ** self.rho

        noise = torch.randn_like(x_start)
        
        x_t = x_start + noise * t.view(-1, 1)
        x_t2 = x_start + noise * t2.view(-1, 1)

        pred_t = model(x_t, t)
        pred_t2 = model(x_t2, t2)

        loss = F.mse_loss(pred_t, pred_t2)
        
        return loss
    
    @torch.no_grad()
    def sample(self, model, shape, device, steps=40):
        x = torch.randn(*shape, device=device) * self.sigma_max

        timesteps = self.get_timesteps(steps).to(device)

        x = model(x, timesteps[0].expand(shape[0]))
        
        return x
    
    @torch.no_grad()
    def multistep_sample(self, model, shape, device, steps=40, num_sampling_steps=2):
        x = torch.randn(*shape, device=device) * self.sigma_max

        timesteps = self.get_timesteps(steps).to(device)

        sampling_indices = torch.linspace(0, steps-1, num_sampling_steps).long()
        sampling_timesteps = timesteps[sampling_indices]

        x = model(x, sampling_timesteps[0].expand(shape[0]))

        for i in range(1, num_sampling_steps):
            noise = torch.randn_like(x)
            t = sampling_timesteps[i]
            x = x + noise * t.view(-1, 1)

            x = model(x, t.expand(shape[0]))
        
        return x
