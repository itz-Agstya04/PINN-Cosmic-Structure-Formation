import torch


def poisson_residual(model, xvt, G=1.0, rho_bar=0.0):
    """
    Computes Poisson equation residual in 1D.

    ∂²φ/∂x² = 4πG (ρ - ρ̄)

    xvt : tensor (N, 3) -> [x, v, t]
    """

    xvt.requires_grad_(True)

    f, phi = model(xvt)

    # First derivative of phi w.r.t x
    grad_phi = torch.autograd.grad(
        phi,
        xvt,
        grad_outputs=torch.ones_like(phi),
        create_graph=True
    )[0]

    phi_x = grad_phi[:, 0:1]

    # Second derivative of phi w.r.t x
    grad_phi_x = torch.autograd.grad(
        phi_x,
        xvt,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True
    )[0]

    phi_xx = grad_phi_x[:, 0:1]

    # Approximate density rho by integrating f over velocity
    # (mean is used here for simplicity)
    rho = f.mean()

    residual = phi_xx - 4.0 * torch.pi * G * (rho - rho_bar)

    return residual
