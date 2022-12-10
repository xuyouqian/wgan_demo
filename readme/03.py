import torch

a = torch.tensor([[1, 7, 3], [4, 8, 6],[7,9,9]],dtype=torch.float32)
model = torch.nn.BatchNorm1d(3)

out = model(a)
print(out)
