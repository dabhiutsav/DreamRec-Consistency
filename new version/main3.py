import os
import time as Time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

def calculate_hit(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)  # (B, C, T)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / (self.hidden_size ** 0.5)
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output = torch.where(torch.eq(key_mask, 0), key_paddings, matmul_output)
        # Causality (if needed)
        diag_vals = torch.ones_like(matmul_output[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        matmul_output = torch.where(torch.eq(causality_mask, 0), key_paddings, matmul_output)
        attn = self.softmax(matmul_output)
        attn = self.dropout(attn)
        output_ws = torch.bmm(attn, V_)
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)
        output = output + queries
        return output

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
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1, sigma_data=0.5):
        super(ConsistencyModel, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.sigma_data = sigma_data
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        
        # None embedding for unconditional generation
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        
        # Positional embeddings
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )

        # Transformer layers for sequence encoding
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        
        # Time step embedding
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )
        
        # Consistency model network - ensure output dimension matches input dimension
        self.network = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, x, h, t):
        """
        Forward pass with consistency model parameterization
        
        Args:
            x: Input tensor (noisy item embedding)
            h: History encoding
            t: Time step
        """
        # Ensure proper dimensions
        if t.dim() == 0:
            t = t.view(1)
        
        # Get time step embedding
        t_emb = self.step_mlp(t)
        
        # Ensure t_emb has the correct batch dimension
        if t_emb.shape[0] != x.shape[0]:
            t_emb = t_emb.expand(x.shape[0], -1)
        
        # Ensure h has the correct batch dimension
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.shape[0] != x.shape[0]:
            h = h.expand(x.shape[0], -1)
        
        # Apply network
        combined = torch.cat((x, h, t_emb), dim=1)
        model_output = self.network(combined)
        
        # Apply skip connection with proper scaling (boundary condition)
        c_skip = self.sigma_data**2 / (t**2 + self.sigma_data**2)
        c_out = t * self.sigma_data / (t**2 + self.sigma_data**2)**0.5
        
        # Ensure proper broadcasting
        c_skip = c_skip.view(-1, 1)
        c_out = c_out.view(-1, 1)
        
        return c_skip * x + c_out * model_output

    def forward_uncon(self, x, t):
        """Unconditional forward pass"""
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = h.expand(x.shape[0], -1)
        
        return self.forward(x, h, t)

    def cacu_x(self, x):
        """Embed items"""
        x = self.item_embeddings(x)
        return x

    def cacu_h(self, states, len_states, p):
        """Encode interaction history with transformer"""
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        # Apply dropout mask for classifier-free guidance
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)

        return h
    
    def predict(self, states, len_states):
        """Generate recommendations using one-step consistency sampling"""
        # Encode history
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()
        
        # Sample Gaussian noise
        x = torch.randn_like(h) * 80.0
        
        # Create time tensor with proper batch dimension
        batch_size = x.shape[0]
        t = torch.full((batch_size,), 80.0, device=self.device)
        
        # One-step denoising with consistency model
        x = self.forward(x, h, t)
        
        # Calculate scores
        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        
        return scores

    
class ConsistencyTraining:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def get_timesteps(self, num_steps):
        """Get discretized time steps for consistency training"""
        steps = torch.arange(num_steps, dtype=torch.float)
        sigma_min_rho = self.sigma_min**(1/self.rho)
        sigma_max_rho = self.sigma_max**(1/self.rho)
        sigmas = (sigma_max_rho + steps/(num_steps-1) * (sigma_min_rho - sigma_max_rho))**self.rho
        return sigmas
    
    def consistency_loss(self, model, x_start, h, device, num_steps=100):
        """Compute consistency training loss"""
        # Sample two adjacent time steps
        n = torch.randint(0, num_steps-1, (x_start.shape[0],), device=device)
        sigmas = self.get_timesteps(num_steps).to(device)
        t_n = sigmas[n]
        t_n_plus_1 = sigmas[n+1]
        
        # Sample noise
        z = torch.randn_like(x_start)
        
        # Add noise at two adjacent time steps
        x_t_n = x_start + t_n.view(-1, 1) * z
        x_t_n_plus_1 = x_start + t_n_plus_1.view(-1, 1) * z
        
        # Compute model outputs
        pred_n = model(x_t_n, h, t_n.view(-1))
        
        with torch.no_grad():
            pred_n_plus_1 = model(x_t_n_plus_1, h, t_n_plus_1.view(-1))
        
        # Compute loss
        loss = F.mse_loss(pred_n, pred_n_plus_1)
        
        return loss

def evaluate(model, test_data, device):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)

    num_total = len(seq)
    topk = [10, 20, 50]

    for i in range(num_total // batch_size):
        seq_b = seq[i * batch_size: (i + 1) * batch_size]
        len_seq_b = len_seq[i * batch_size: (i + 1) * batch_size]
        target_b = target[i * batch_size: (i + 1) * batch_size]
        
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(device)

        prediction = model.predict(states, np.array(len_seq_b))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase)
        total_purchase += batch_size
 
    hr_list = []
    ndcg_list = []
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 
        'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 
        'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i]/total_purchase
        ng_purchase = ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])

        if i == 1:
            hr_20 = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))

    return hr_20

if __name__ == '__main__':
    data_directory = '.\data\yc'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    
    seq_size = data_statis['seq_size'][0]
    item_num = data_statis['item_num'][0]

    # Initialize model and training components
    model = ConsistencyModel(hidden_size=64, item_num=item_num, state_size=seq_size, dropout=0.1, device=device, sigma_data=0.5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, eps=1e-8, weight_decay=0.0)
    
    # Initialize consistency training
    ct = ConsistencyTraining(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    
    # Load training data
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    
    # Training loop
    hr_max = 0
    best_epoch = 0
    num_rows = train_data.shape[0]
    num_batches = int(num_rows/256)
    
    for i in range(200):
        start_time = Time.time()
        
        for j in range(num_batches):
            batch = train_data.sample(n=256).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()
            seq = torch.LongTensor(seq).to(device)
            len_seq = torch.LongTensor(len_seq).to(device)
            target = torch.LongTensor(target).to(device)

            # Get item embeddings and history encoding
            x_start = model.cacu_x(target)
            h = model.cacu_h(seq, len_seq, 0.1)
            
            # Compute consistency training loss
            loss = ct.consistency_loss(model, x_start, h, device, num_steps=100)
            
            loss.backward()
            optimizer.step()

        if True:
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(loss) + 
                      "Time cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-start_time)))

            if (i + 1) % 2 == 0:
                eval_start = Time.time()
                print('-------------------------- VAL PHASE --------------------------')
                hr = evaluate(model, 'val_data.df', device)
                print('-------------------------- TEST PHASE -------------------------')
                _ = evaluate(model, 'test_data.df', device)
                print("Evaluation cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                print('----------------------------------------------------------------')

                if hr > hr_max:
                    hr_max = hr
                    best_epoch = i
    
    print(f"Best epoch: {best_epoch}, Best HR@20: {hr_max}")
