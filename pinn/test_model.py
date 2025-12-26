import torch
from pinn.Model import PINN

model = PINN()
x = torch.randn(10, 3)

f, phi = model(x)

print(f.shape)
print(phi.shape)
