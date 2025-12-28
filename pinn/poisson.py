import torch


def poisson_residual(model, xvt, G=1.0):
    xvt = xvt.clone().detach().requires_grad_(True)

    _, phi = model(xvt)

    grad_phi = torch.autograd.grad(
        phi, xvt,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True
    )[0]

    phi_x = grad_phi[:, 0:1]

    grad_phi_x = torch.autograd.grad(
        phi_x, xvt,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True,
        retain_graph=True
    )[0]

    phi_xx = grad_phi_x[:, 0:1]

    f, _ = model(xvt)
    rho = f.mean()

    return phi_xx - 4.0 * torch.pi * G * rho
