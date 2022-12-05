import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
import sys

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    return data.values

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
    def forward(self):
        pred = torch.matmul(self.user_emb.weight,self.item_emb.weight.t())
        return pred


if __name__ == "__main__":

    data_source = sys.argv[1]
    ratings = load_data(data_source)
    model = MF(num_users=ratings.shape[0], num_items=ratings.shape[1], embedding_dim=20)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),  weight_decay=0.001, lr=0.01)

    for i in range(100):
        ratings = torch.tensor(ratings, dtype=torch.float)
        preds = model()
        loss = loss_fn(preds, ratings)
        print(loss.mean().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    domain = "s" if data_source.split(".")[-1]=="s.csv" else "t"
    torch.save(model.state_dict(), f"MF_{domain}.pt")