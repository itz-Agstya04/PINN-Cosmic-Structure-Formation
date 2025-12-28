import torch


def vlasov_residual(model, xvt):
    xvt = xvt.clone().detach().requires_grad_(True)

    f, phi = model(xvt)

    grad_f = torch.autograd.grad(
        f, xvt,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True
    )[0]

    f_x = grad_f[:, 0:1]
    f_v = grad_f[:, 1:2]
    f_t = grad_f[:, 2:3]

    grad_phi = torch.autograd.grad(
        phi, xvt,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True
    )[0]

    phi_x = grad_phi[:, 0:1]

    v = xvt[:, 1:2]
    return f_t + v * f_x - phi_x * f_v
