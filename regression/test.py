import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import regression

device = torch.device("cuda")

df = pd.read_csv("covid_test.csv", header=0)
df = df.to_numpy(dtype = np.float32)
data = df[:,1:]
data = torch.from_numpy(data).to(device)

model = regression(87, 1).to(device)
model.load_state_dict(torch.load("pt/regression_299.pt"))
model.eval()

pred = []

for i in range(len(data)): pred.append(model(data[i]).item())

result = pd.DataFrame(pred)
result.to_csv("result299.csv", header = ["test_positive"], index = None)