import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
from MF import MF
from MLP import MLP
from BPR import BPRModel

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    return data.values

data_source = "data/t_rate.csv"
ratings = load_data(data_source)

n_users = ratings.shape[0]

mapping_model = MLP(20,2)
mapping_model.eval()
mapping_model.load_state_dict(torch.load("MLP.pt"))


s_model = MF(ratings.shape[0], ratings.shape[1], 20)
s_model.load_state_dict(torch.load(f"MF_s.pt"))

t_model =MF(ratings.shape[0], ratings.shape[1], 20)
t_model.load_state_dict(torch.load(f"MF_t.pt"))

source_embeddings = s_model.user_emb.weight.detach()
target_embeddings = mapping_model(source_embeddings)
t_model.user_emb.weight = nn.Parameter(target_embeddings)


loss_fn = nn.MSELoss()
with torch.no_grad():
    preds = t_model()
    loss = loss_fn(preds, torch.tensor(ratings))
    print(loss.item())