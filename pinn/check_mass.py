import torch
import numpy as np

from pinn.model import PINN
from pinn.mass import mass_over_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN().to(device)
    model.load_state_dict(torch.load("results/model_lbfgs.pt", map_location=device))
    model.eval()

    # Time points to check
    t_values = np.linspace(0.0, 1.0, 10)

    masses = mass_over_time(model, device, t_values)

    for t, m in zip(t_values, masses):
        print(f"t = {t:.2f} | M(t) = {m:.6f}")

    # Save results
    np.save("results/mass_history.npy", masses)
    print("Mass history saved to results/mass_history.npy")


if __name__ == "__main__":
    main()
