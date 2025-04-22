# src/lightgcn_trainer.py
"""
LightGCN training module: includes dataset preparation, adjacency construction, model definition, and BPR training loop.
"""
import os
import json
import re
import random
import numpy as np
torch_available = True
try:
    import torch
    import torch.nn as nn
    from scipy.sparse import coo_matrix
except ImportError:
    torch_available = False


def load_interactions(json_path, user_offset=0):
    """
    Load user-book interactions from a JSON file.
    Each entry should contain an 'input' field with book titles in quotes.
    Returns list of (user_index, title) and number of users.
    """
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    interactions = []
    for idx, entry in enumerate(data):
        user_id = idx + user_offset
        titles = re.findall(r'"([^\"]+)"', entry.get('input', ''))
        titles += re.findall(r'"([^\"]+)"', entry.get('output', ''))
        for title in titles:
            interactions.append((user_id, title.strip()))
    interactions = list(set(interactions))
    return interactions, len(data)


def build_norm_adj(interactions, num_users, num_items):
    """
    Construct symmetric normalized adjacency matrix for LightGCN.
    Returns a torch sparse tensor.
    """
    rows, cols, vals = [], [], []
    for u, i in interactions:
        rows += [u, i + num_users]
        cols += [i + num_users, u]
        vals += [1.0, 1.0]
    N = num_users + num_items
    A = coo_matrix((vals, (rows, cols)), shape=(N, N))
    deg = np.array(A.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    D = coo_matrix((deg_inv_sqrt, (np.arange(N), np.arange(N))), shape=(N, N))
    A_norm = D.dot(A).dot(D).tocoo()
    indices = torch.LongTensor(np.vstack((A_norm.row, A_norm.col)))
    values = torch.FloatTensor(A_norm.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(A_norm.shape))


class LightGCN(nn.Module):
    """
    LightGCN model: learns user and item embeddings via graph propagation.
    """
    def __init__(self, num_users, num_items, emb_dim, n_layers, adj):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.adj = adj
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            all_emb.append(x)
        stacked = torch.stack(all_emb, dim=1)
        final_emb = stacked.mean(dim=1)
        return final_emb[:self.num_users], final_emb[self.num_users:]


def bpr_loss(user_emb, pos_emb, neg_emb):
    """
    Compute Bayesian Personalized Ranking loss for one batch.
    """
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    return loss


def lightgcn_trainer(config):
    """
    Train LightGCN using BPR on Goodreads interactions.
    """
    # Unpack config
    data_dir = config['data_dir']
    train_json = config['train_json']
    valid_json = config['valid_json']
    model_filename = config['model_filename']
    emb_dim = config['embedding_dim']
    n_layers = config['num_layers']
    lr = float(config['learning_rate'])
    epochs = config['epochs']
    bs = config['batch_size']
    seed = config.get('seed', 42)

    # Prepare output directory
    model_dir = os.path.join(data_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_filename)

    # Load interactions
    train_inter, n_train = load_interactions(train_json, user_offset=0)
    valid_inter, n_valid = load_interactions(valid_json, user_offset=n_train)
    all_inter = train_inter + valid_inter
    num_users = n_train + n_valid
    all_books = {title for _, title in all_inter}
    book2idx = {b:i for i, b in enumerate(sorted(all_books))}
    num_items = len(book2idx)
    data_idx = [(u, book2idx[t]) for u, t in all_inter]
    inter_set = set(data_idx)

    # Build adjacency
    adj = build_norm_adj(data_idx, num_users, num_items).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightGCN(num_users, num_items, emb_dim, n_layers, adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    random.seed(seed)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, count = 0.0, 0
        random.shuffle(data_idx)
        for start in range(0, len(data_idx), bs):
            batch = data_idx[start:start+bs]
            users = torch.LongTensor([u for u, _ in batch]).to(device)
            pos = torch.LongTensor([i for _, i in batch]).to(device)
            # sample negatives
            neg = []
            for u in users.cpu().tolist():
                ni = random.randrange(num_items)
                while (u, ni) in inter_set:
                    ni = random.randrange(num_items)
                neg.append(ni)
            neg = torch.LongTensor(neg).to(device)

            ue, ie = model()
            loss = bpr_loss(ue[users], ie[pos], ie[neg])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            count += len(batch)
        print(f"Epoch {epoch}/{epochs}, BPR Loss: {total_loss/count:.4f}")

    # Save model state
    torch.save(model.state_dict(), save_path)
    print(f"LightGCN model saved to {save_path}")
