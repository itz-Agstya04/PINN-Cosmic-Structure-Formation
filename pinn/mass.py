import torch
import numpy as np


@torch.no_grad()
def compute_mass(model, device, t_value, N=20000):
    """
    Computes total mass M(t) = ∫ f(x, v, t) dx dv
    using Monte Carlo integration.
    """

    model.eval()

    # Sample phase space
    x = torch.rand(N, 1, device=device) * 2 - 1
    v = torch.rand(N, 1, device=device) * 2 - 1
    t = torch.full((N, 1), t_value, device=device)

    xvt = torch.cat([x, v, t], dim=1)

    f, _ = model(xvt)

    # Phase-space volume = (2 * 2) = 4
    volume = 4.0

    mass = volume * torch.mean(f).item()
    return mass


def mass_over_time(model, device, t_values, N=20000):
    """
    Computes mass at multiple time points.
    """

    masses = []
    for t in t_values:
        m = compute_mass(model, device, t, N)
        masses.append(m)

    return np.array(masses)
