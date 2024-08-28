import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class covid19Data(Dataset):
    def __init__(self, df):
        super(covid19Data, self).__init__()
        self.x = df[:,1:-1]
        self.y = df[:,-1].reshape(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__=="__main__":
    df = pd.read_csv("covid_train.csv", header=0)
    print(df)

    df = df.to_numpy(dtype = np.float32)
    df = torch.from_numpy(df)
    train_data = covid19Data(df)

    print(train_data.x[0].size())