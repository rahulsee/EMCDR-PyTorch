import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
import sys

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data.index-=1 
    data = data.fillna(0)
    users = data.index
    movies = data.columns
    
    user_ratings = defaultdict(dict)
    no = 0
    for i in range(len(users)):
        user_movies = list(np.where(data.loc[users[i]] != 0.0)[0])
        if len(user_movies) > 1:
            user_ratings[no] = user_movies
            no += 1
    
    movies = [int(movie) for movie in movies]
    return users, movies, user_ratings

def generate_test(user_ratings):
    user_ratings_test = {}
    for user in user_ratings:
        user_ratings_test[user] = random.sample(user_ratings[user], 1)[0]
    return user_ratings_test

def generate_train_batch(user_ratings, user_ratings_test, n, batch_size=512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(0, n-1)
        while j in user_ratings[u]:
            j = random.randint(0, n-1)
        
        t.append([u, i, j])
    
    train_batch = np.asarray(t)
    return train_batch

def generate_test_batch(user_ratings, user_ratings_test, n):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(n):
            if j not in user_ratings[u]:
                t.append([u, i, j])
        # print(t)
        yield np.asarray(t)

class BPRModel(nn.Module):
  def __init__(self, n_users, n_items, embedding_dim):
    super(BPRModel, self).__init__()
    self.n_users = n_users
    self.n_items = n_items
    self.user_emb = nn.Embedding(n_users, embedding_dim)
    torch.nn.init.xavier_uniform_(self.user_emb.weight)
    self.item_emb = nn.Embedding(n_items, embedding_dim)
    torch.nn.init.xavier_uniform_(self.item_emb.weight)
  def forward(self, user_ids, item_ids_1, item_ids_2):
    user_embs = self.user_emb(user_ids)
    item_embs_1 = self.item_emb(item_ids_1)
    item_embs_2 = self.item_emb(item_ids_2)
    return user_embs, item_embs_1, item_embs_2

class BPRLoss(nn.Module):
  def __init__(self):
    super(BPRLoss, self).__init__()
    self.sigmoid = nn.Sigmoid()
  def forward(self, user_embs, item_embs_1, item_embs_2):
    diff = (item_embs_1 - item_embs_2).unsqueeze(2)
    user_embs = user_embs.unsqueeze(1)
    sim = torch.bmm(user_embs, diff).squeeze(2)
    sim = -torch.log(self.sigmoid(sim))
    return sim


if __name__ == '__main__':

    data_source = sys.argv[1]

    users, movies, user_ratings = load_data(data_source)
    user_ratings_test = generate_test(user_ratings)

    model = BPRModel(len(users), len(movies), embedding_dim=20)
    loss_fn = BPRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    for i in range(100):
        batch = generate_train_batch(user_ratings, user_ratings_test, len(movies), batch_size=512)
        u, i1, i2 = model(torch.tensor(batch[:, 0]), torch.tensor(batch[:, 1]), torch.tensor(batch[:, 2]))
        loss = loss_fn(u, i1, i2)
        if i%10 == 0:
            print(f"EPOCH: {i} LOSS: {loss.mean().item()}")
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if i%10 == 0:
            with torch.no_grad():
                test_loss=0
                length = 0
                for batch in generate_test_batch(user_ratings, user_ratings_test, len(movies)):
                    u_t, i1_t, i2_t = model(torch.tensor(batch[:, 0]), torch.tensor(batch[:, 1]), torch.tensor(batch[:, 2]))
                    loss = loss_fn(u_t, i1_t, i2_t)
                    test_loss += loss.sum().item()
                    length += loss.shape[0]
                print(f"TEST LOSS is {test_loss/length}")

    domain = "s" if data_source.split("/")[-1]=="s_rate.csv" else "t"

    torch.save(model.state_dict(), f"BPR_{domain}.pt")
    