import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),  # input: (x, v, t)
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 2)   # output: (f, phi)
        )

        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: (N, 3) -> [x, v, t]
        returns:
            f   : phase-space density
            phi : gravitational potential
        """
        out = self.net(x)

        f = out[:, 0:1]
        phi = out[:, 1:2]

        return f, phi

