import os
import torch
import csv
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocessing
from copy import deepcopy as dc
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]



def create_dataloader(csv_file_path, batch_size, lookback=60):
    X, y = preprocessing(csv_file_path, lookback)
    X = dc(np.flip(X, axis=1))
    
    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    batch = next(iter(test_loader))
    print(batch[1].shape)

    return train_loader, test_loader


if __name__ == "__main__":
    create_dataloader("./HONAUT.NS_data.csv", 1)