import numpy as np
import pandas as pd
import random
import time
import torch
from torch import nn
import os
from utility import calculate_hit, extract_axis_1
from modules import *
from consistency_model import ConsistencyModel, ConsistencyTraining

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, model_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.model_type = model_type
        self.device = device

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)

        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.consistency_model = ConsistencyModel(hidden_size)

    def forward(self, x, t):
        return self.consistency_model(x, t)
    
    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def cacu_h(self, states, len_states, p):
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

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)
        
        return h
    
    def predict(self, states, len_states, consistency_trainer, sampling_steps=1):
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

        batch_size = h.shape[0]
        shape = (batch_size, self.hidden_size)
        
        if sampling_steps == 1:
            x = consistency_trainer.sample(self, shape, self.device)
        else:
            x = consistency_trainer.multistep_sample(self, shape, self.device, num_sampling_steps=sampling_steps)

        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        
        return scores

def evaluate(model, test_data, consistency_trainer, device, sampling_steps=1):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    topk = [10, 20, 30]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)

    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.predict(states, np.array(len_seq_b), consistency_trainer, sampling_steps)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase)

        total_purchase += batch_size

    hr_list = []
    ndcg_list = []
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i]/total_purchase
        ng_purchase = ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)

        if i == 1:
            hr_20 = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1], hr_list[2], ndcg_list[2]))

    return hr_20


if __name__ == '__main__':
    hidden_factor = 64
    dropout_rate = 0.1
    p = 0.1
    random_seed = 100
    batch_size = 256
    lr = 0.005
    l2_decay = 0
    model_type = 'mlp1'
    epochs = 1000
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    sigma_data = 0.5
    sampling_steps = 1
    
    setup_seed(random_seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_directory = './data/yc'
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    seq_size = data_statis['seq_size'][0]
    item_num = data_statis['item_num'][0]
    topk = [10, 20, 50]
    
    model = Tenc(hidden_factor, item_num, seq_size, dropout_rate, model_type, device)
    consistency_trainer = ConsistencyTraining(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        sigma_data=sigma_data
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=l2_decay)
    
    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    hr_max = 0
    best_epoch = 0
    
    num_rows = train_data.shape[0]
    num_batches = int(num_rows / batch_size)
    
    for i in range(epochs):
        start_time = time.time()
        
        for j in range(num_batches):
            batch = train_data.sample(n=batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())
            
            optimizer.zero_grad()
            
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            
            x_start = model.cacu_x(target)
            
            h = model.cacu_h(seq, len_seq, p)
            
            loss = consistency_trainer.consistency_loss(model, x_start, h)

            loss.backward()
            optimizer.step()
        
        if i % 1 == 0:
            print(f"Epoch {i:03d}; Train loss: {loss:.4f}; Time cost: {time.time() - start_time:.2f}s")
        
        if (i + 1) % 10 == 0:
            eval_start = time.time()
            print('-------------------------- VAL PHRASE --------------------------')
            hr = evaluate(model, 'val_data.df', consistency_trainer, device, sampling_steps)
            print('-------------------------- TEST PHRASE -------------------------')
            _ = evaluate(model, 'test_data.df', consistency_trainer, device, sampling_steps)
            print(f"Evaluation cost: {time.time() - eval_start:.2f}s")
            print('----------------------------------------------------------------')
            
            if hr > hr_max:
                hr_max = hr
                best_epoch = i
                torch.save(model.state_dict(), f'best_model__{model_type}.pt')
    
    print(f"Best epoch: {best_epoch}, Best HR@20: {hr_max}")
