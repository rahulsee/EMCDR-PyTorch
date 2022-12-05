import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
from MF import MF

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    return data.values

data_source = "data/s_rate.csv"
ratings = load_data(data_source)

n_users = ratings.shape[0]

class MLP(nn.Module):
    def __init__(self, dim, layers):
        super(MLP, self).__init__()
        self.hidden = 128
        self.layers = layers - 1
        self.net_in = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
        )
        self.net_hid = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.net_out = nn.Sequential(
            nn.Linear(64, dim)
        )

    def forward(self, u):
        x = self.net_in(u)
        for _ in range(self.layers):
            x = self.net_hid(x)
        x = self.net_out(x)
        return x


if __name__ == "__main__":

    model = MLP(20, 2)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.001)

    s_model = MF(ratings.shape[0], ratings.shape[1], 20)
    s_model.load_state_dict(torch.load(f"MF_s.pt"))

    t_model = MF(ratings.shape[0], ratings.shape[1], 20)
    t_model.load_state_dict(torch.load(f"MF_t.pt"))

    target = t_model.user_emb.weight.detach()
    source = s_model.user_emb.weight.detach()

    for i in range(100):
        pred = model(source)
        loss = loss_fn(pred, target)
        print(loss.item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    torch.save(model.state_dict(), "MLP.pt")