import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn.model import PINN


# --------------------------------------------------
# LOSS PLOT
# --------------------------------------------------
def plot_losses():
    loss_history = torch.load("results/loss_history.pt")

    plt.figure()
    plt.semilogy(loss_history["total"], label="Total")
    plt.semilogy(loss_history["vlasov"], label="Vlasov")
    plt.semilogy(loss_history["poisson"], label="Poisson")
    plt.semilogy(loss_history["ic"], label="IC")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses (Adam)")

    plt.tight_layout()
    plt.savefig("results/loss_plot.png")
    plt.show()


# --------------------------------------------------
# PHASE SPACE PLOT
# --------------------------------------------------
@torch.no_grad()
def plot_phase_space(t_value=1.0, N=30000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN().to(device)
    model.load_state_dict(torch.load("results/model_lbfgs.pt", map_location=device))
    model.eval()

    x = torch.rand(N, 1, device=device) * 2 - 1
    v = torch.rand(N, 1, device=device) * 2 - 1
    t = torch.full((N, 1), t_value, device=device)

    xvt = torch.cat([x, v, t], dim=1)
    f, _ = model(xvt)

    plt.figure()
    plt.scatter(
        x.cpu().numpy(),
        v.cpu().numpy(),
        c=f.cpu().numpy(),
        s=1,
        cmap="inferno"
    )
    plt.colorbar(label="f(x, v)")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(f"Phase Space Density at t = {t_value}")

    plt.tight_layout()
    plt.savefig("results/phase_space.png")
    plt.show()


# --------------------------------------------------
# MASS PLOT
# --------------------------------------------------
def plot_mass():
    masses = np.load("results/mass_history.npy")
    t_values = np.linspace(0.0, 1.0, len(masses))

    plt.figure()
    plt.plot(t_values, masses, marker="o")
    plt.xlabel("Time")
    plt.ylabel("Total Mass")
    plt.title("Mass Conservation Check")

    plt.tight_layout()
    plt.savefig("results/mass_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_losses()
    plot_phase_space(t_value=1.0)
    plot_mass()
