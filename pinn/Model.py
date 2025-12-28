import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

        self.softplus = nn.Softplus(beta=5.0)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)

        # Enforce f >= 0 (physical constraint)
        f = self.softplus(out[:, 0:1])

        # Gravitational potential can be unconstrained
        phi = out[:, 1:2]

        return f, phi
