#!/usr/bin/env python
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import time

# ----------------------------
# Utility Functions
# ----------------------------
def extract(a, t, x_shape):
    """
    Extract values from a tensor a at indices t and reshape to match x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    Linear schedule for beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

# ----------------------------
# Consistency Model (Replacing Diffusion)
# ----------------------------
class Consistency:
    def __init__(self, timesteps, beta_start, beta_end, lambda_cons=1.0):
        self.timesteps = timesteps
        self.lambda_cons = lambda_cons
        # Create beta schedule and compute derived quantities
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward (noising) process: given clean x_start and noise level t,
        return the noisy version.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_ac = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_one_ac * noise

    def consistency_loss(self, model, x_start, h, t, noise=None):
        """
        Compute the loss for the consistency model.
        Loss = Reconstruction loss + λ * Consistency loss,
        where:
          - Reconstruction loss: MSE(model(x_noisy, t, h) , x_start)
          - Consistency loss: MSE(model(model(x_noisy, t, h), s, h) , model(x_noisy, t, h))
        Here we use s = 0 (the clean condition) for the second pass.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x0_pred = model(x_noisy, h, t)
        recon_loss = F.mse_loss(x0_pred, x_start)
        s = torch.zeros_like(t)  # use zero noise level for consistency check
        x0_pred_re = model(x0_pred, h, s)
        cons_loss = F.mse_loss(x0_pred_re, x0_pred)
        return recon_loss + self.lambda_cons * cons_loss

    @torch.no_grad()
    def sample(self, model, model_uncond, h, device):
        """
        One-step sampling: start from a noisy sample at the highest noise level
        and obtain the denoised (oracle) item embedding.
        """
        t = torch.full((h.shape[0],), self.timesteps - 1, dtype=torch.long, device=device)
        # Sample noise such that x_t = sqrt(1 - ᾱ_t) * noise (assume x0=0)
        noise = torch.randn(h.shape[0], model.hidden_size, device=device)
        sqrt_one_ac = extract(self.sqrt_one_minus_alphas_cumprod, t, (h.shape[0], model.hidden_size))
        x = sqrt_one_ac * noise
        x0_pred = model(x, h, t)
        return x0_pred

# ----------------------------
# Additional Modules (Adapted from DreamRec’s original Modules_ori)
# ----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
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
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(split_size)
        attn = self.softmax(matmul_output)
        attn = self.dropout(attn)
        output_ws = torch.bmm(attn, V_)
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)
        return output + queries

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

# ----------------------------
# Tenc: The DreamRec Model
# ----------------------------
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)

        self.none_embedding = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)

        self.emb_dropout = nn.Dropout(dropout)

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)

        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

        self.s_fc = nn.Linear(hidden_size, item_num)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        if diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size))
        elif diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        else:
            self.diffuser = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size))
    
    def forward(self, x, h, step):
        # x: [B, hidden_size], h: [B, hidden_size]
        # step: [B] -> unsqueeze to [B, 1] and pass through step_mlp
        t = self.step_mlp(step.float().unsqueeze(1)).squeeze(1)  # Now shape is [B, hidden_size]
        combined = torch.cat((x, h, t), dim=1)
        out = self.diffuser(combined)
        return out

    
    def forward_uncond(self, x, step):
        """
        Unconditional branch: use a dummy guidance embedding.
        """
        B = x.shape[0]
        h = self.none_embedding(torch.zeros(B, dtype=torch.long, device=self.device))
        t = self.step_mlp(step.float().unsqueeze(1))
        combined = torch.cat((x, h, t), dim=1)
        out = self.diffuser(combined)
        return out

    def cacu_x(self, x):
        """
        Compute the target item embedding given its index.
        """
        return self.item_embeddings(x)
    
    def cacu_h(self, states, len_states, p):
        """
        Compute the guidance embedding from the historical interaction sequence.
        """
        inputs_emb = self.item_embeddings(states)
        positions = torch.arange(self.state_size, device=self.device).unsqueeze(0).expand(inputs_emb.size(0), -1)
        inputs_emb += self.positional_embeddings(positions)
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze(1)
        B, D = h.shape
        mask1d = (torch.sign(torch.rand(B, device=self.device) - p) + 1) / 2
        mask1d = mask1d.view(B, 1)
        h = h * mask1d + self.none_embedding(torch.zeros(B, dtype=torch.long, device=self.device)).squeeze(1) * (1 - mask1d)
        return h  

    def predict(self, states, len_states, cons_model):
        """
        Generate the oracle item embedding via one-step sampling and then
        compute scores over all item embeddings.
        """
        inputs_emb = self.item_embeddings(states)
        positions = torch.arange(self.state_size, device=self.device).unsqueeze(0).expand(inputs_emb.size(0), -1)
        inputs_emb += self.positional_embeddings(positions)
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze(1)
        x = cons_model.sample(self, self.forward_uncond, h, self.device)
        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0,1))
        return scores

# ----------------------------
# Training and Evaluation Functions
# ----------------------------
def train(model, cons_model, optimizer, data_directory, device, args):
    """
    Train the model using the consistency loss.
    Assumes training data is stored in a pickled dataframe 'train.df'
    with columns: 'seq' (historical sequence), 'len_seq', and 'next' (target item).
    """
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    model.train()
    num_samples = len(train_data)
    batch_size = args.batch_size
    for epoch in range(args.epoch):
        start_time = time.time()

        epoch_loss = 0.0
        indices = list(range(num_samples))
        random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = train_data.iloc[batch_indices]
            seq_batch = np.stack(batch['seq'].values)
            len_seq_batch = np.array(batch['len_seq'].values)
            next_batch = np.array(batch['next'].values)
            states = torch.LongTensor(seq_batch).to(device)
            # Compute guidance representation from history
            h = model.cacu_h(states, len_seq_batch, args.p)
            # Get target item embeddings
            next_batch = torch.LongTensor(next_batch).to(device)
            x0 = model.cacu_x(next_batch)
            # Sample noise level t uniformly from [0, cons_model.timesteps - 1]
            t = torch.randint(0, cons_model.timesteps, (x0.shape[0],), device=device)
            loss = cons_model.consistency_loss(model.forward, x0, h, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (num_samples / batch_size)
        print(f"Epoch {epoch+1}/{args.epoch}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        if (epoch + 1) % 10 == 0:
            eval_start = time.time()
            print('\n-------------------------- VAL PHASE --------------------------')
            _ = evaluate_model(model, cons_model, 'val_data.df', data_directory, device)
            print('-------------------------- TEST PHASE -------------------------')
            _ = evaluate_model(model, cons_model, 'test_data.df', data_directory, device)
            print(f"Evaluation cost: {time.time() - eval_start:.2f}s")
            print('----------------------------------------------------------------\n')


def evaluate_model(model, cons_model, test_file, data_directory, device, topk=[10,20,50]):
    """
    Evaluate the model on test data (assumed to be in 'test.df').
    For each test instance, generate an oracle item embedding and retrieve the top-K nearest items.
    """
    test_df = pd.read_pickle(os.path.join(data_directory, test_file))
    total_samples = len(test_df)
    hit = [0] * len(topk)
    ndcg = [0] * len(topk)
    batch_size = 100
    for i in range(0, total_samples, batch_size):
        batch = test_df.iloc[i:i+batch_size]
        seq_batch = np.stack(batch['seq'].values)
        len_seq_batch = np.array(batch['len_seq'].values)
        target_batch = np.array(batch['next'].values)
        states = torch.LongTensor(seq_batch).to(device)
        scores = model.predict(states, len_seq_batch, cons_model)
        _, topK_indices = scores.topk(100, dim=1, largest=True, sorted=True)
        topK_indices = topK_indices.cpu().detach().numpy()
        sorted_list = np.flip(topK_indices, axis=1)
        for j in range(len(target_batch)):
            true_item = target_batch[j]
            for k, k_val in enumerate(topk):
                rec_list = sorted_list[j, -k_val:]
                if true_item in rec_list:
                    hit[k] += 1
                    rank = k_val - np.where(rec_list == true_item)[0][0]
                    ndcg[k] += 1.0 / np.log2(rank + 1)
    hr = [h / total_samples for h in hit]
    ndcg = [n / total_samples for n in ndcg]
    
    for k, k_val in enumerate(topk):
        print(f"HR@{k_val}: {hr[k]:.4f}, NDCG@{k_val}: {ndcg[k]:.4f}")

    return hr, ndcg

# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--data', nargs='?', default='yc', help='Data directory name (e.g., yc, ks, zhihu)')
    parser.add_argument('--random_seed', type=int, default=100, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--timesteps', type=int, default=50, help='Number of noise timesteps')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value')
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning rate')
    parser.add_argument('--l2_decay', type=float, default=0, help='L2 regularization')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--w', type=float, default=2.0, help='Guidance strength (not used in consistency version)')
    parser.add_argument('--p', type=float, default=0.1, help='Probability for unconditional training (classifier-free guidance)')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Type of diffuser (mlp1 or mlp2)')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    data_directory = os.path.join('.\data', args.data)
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    seq_size = data_statis['seq_size'][0]
    item_num = data_statis['item_num'][0]
    
    # Instantiate the DreamRec model (Tenc) and the consistency model (replacing diffusion)
    model = Tenc(args.hidden_factor, item_num, seq_size, args.dropout_rate, args.diffuser_type, device).to(device)
    cons_model = Consistency(args.timesteps, args.beta_start, args.beta_end, lambda_cons=1.0)
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    
    train(model, cons_model, optimizer, data_directory, device, args)

if __name__ == '__main__':
    main()
