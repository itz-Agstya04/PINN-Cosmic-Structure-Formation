import torch
import torch.optim as optim

from pinn.model import PINN
from pinn.losses import total_loss


def main():
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = PINN().to(device)
    model.train()

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2000
    N = 1000
    N_ic = 200

    # --------------------------------------------------
    # LOSS HISTORY (IMPORTANT)
    # --------------------------------------------------
    loss_history = {
        "total": [],
        "vlasov": [],
        "poisson": [],
        "ic": []
    }

    # --------------------------------------------------
    # ADAM TRAINING
    # --------------------------------------------------
    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE points
        x = torch.rand(N, 1, device=device) * 2 - 1
        v = torch.rand(N, 1, device=device) * 2 - 1
        t = torch.rand(N, 1, device=device)
        xvt = torch.cat([x, v, t], dim=1)

        # IC points (t = 0)
        x_ic = torch.rand(N_ic, 1, device=device) * 2 - 1
        v_ic = torch.rand(N_ic, 1, device=device) * 2 - 1
        t_ic = torch.zeros(N_ic, 1, device=device)
        xvt_ic = torch.cat([x_ic, v_ic, t_ic], dim=1)

        # Loss
        loss, lv, lp, lic = total_loss(model, xvt, xvt_ic)
        loss.backward()
        optimizer.step()

        # Save losses
        loss_history["total"].append(loss.item())
        loss_history["vlasov"].append(lv.item())
        loss_history["poisson"].append(lp.item())
        loss_history["ic"].append(lic.item())

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Total {loss.item():.3e} | "
                f"V {lv.item():.3e} | "
                f"P {lp.item():.3e} | "
                f"IC {lic.item():.3e}"
            )

    # --------------------------------------------------
    # SAVE ADAM RESULTS
    # --------------------------------------------------
    torch.save(model.state_dict(), "results/model_adam.pt")
    torch.save(loss_history, "results/loss_history.pt")
    print("Adam model + loss history saved.")

    # --------------------------------------------------
    # L-BFGS REFINEMENT
    # --------------------------------------------------
    print("Starting L-BFGS optimization...")

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=500,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()

        x = torch.rand(N, 1, device=device) * 2 - 1
        v = torch.rand(N, 1, device=device) * 2 - 1
        t = torch.rand(N, 1, device=device)
        xvt = torch.cat([x, v, t], dim=1)

        x_ic = torch.rand(N_ic, 1, device=device) * 2 - 1
        v_ic = torch.rand(N_ic, 1, device=device) * 2 - 1
        t_ic = torch.zeros(N_ic, 1, device=device)
        xvt_ic = torch.cat([x_ic, v_ic, t_ic], dim=1)

        loss, _, _, _ = total_loss(model, xvt, xvt_ic)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    torch.save(model.state_dict(), "results/model_lbfgs.pt")
    print("L-BFGS finished. Final model saved.")


if __name__ == "__main__":
    main()
