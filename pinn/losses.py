import torch
from pinn.vlasov import vlasov_residual
from pinn.poisson import poisson_residual


def initial_condition_loss(model, xvt_ic):
    x = xvt_ic[:, 0:1]
    v = xvt_ic[:, 1:2]

    f_pred, _ = model(xvt_ic)
    f_true = torch.exp(-v**2) * (1.0 + 0.1 * torch.cos(2 * torch.pi * x))

    return torch.mean((f_pred - f_true) ** 2)


def total_loss(
    model,
    xvt,
    xvt_ic,
    lambda_vlasov=1.0,
    lambda_poisson=1.0,
    lambda_ic=10.0
):
    r_v = vlasov_residual(model, xvt)
    r_p = poisson_residual(model, xvt)

    loss_v = torch.mean(r_v ** 2)
    loss_p = torch.mean(r_p ** 2)
    loss_ic = initial_condition_loss(model, xvt_ic)

    loss = (
        lambda_vlasov * loss_v +
        lambda_poisson * loss_p +
        lambda_ic * loss_ic
    )

    return loss, loss_v, loss_p, loss_ic
