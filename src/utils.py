# Funcions and classes required for training and stuff
import pickle
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_pkl(path):
    
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def make_loader(X, y, bs=None):

    dataset = TensorDataset(X,y)

    if bs:
        data_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    else:
        data_loader = DataLoader(dataset, shuffle=False)

    return data_loader

def load_model(model, path:str):
    if path.endswith('.pth'):
        w_dir = path
    else:
        w_dir = os.path.join(path, 'test_weight.pth')
    model.load_state_dict(torch.load(w_dir, map_location=device))
    return model







