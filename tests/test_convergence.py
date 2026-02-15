"""Convergence tests: verify LUMA trains small models correctly."""

import torch
import torch.nn as nn

from luma_optimizer import LUMA


def test_rosenbrock(device):
    """Minimise Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2."""
    xy = torch.tensor([-1.0, 1.0], device=device, requires_grad=True)
    opt = LUMA([xy], lr=1e-3, weight_decay=0.0)

    for _ in range(3000):
        opt.zero_grad()
        x, y = xy[0], xy[1]
        loss = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        loss.backward()
        opt.step()

    assert abs(xy[0].item() - 1.0) < 0.5, f"x = {xy[0].item()} on {device}"
    assert abs(xy[1].item() - 1.0) < 0.5, f"y = {xy[1].item()} on {device}"


def test_mlp_classification(device):
    """Train a small MLP on linearly separable synthetic data."""
    n, d = 200, 20
    X = torch.randn(n, d, device=device)
    w_true = torch.randn(d, device=device)
    y = (X @ w_true > 0).float()

    model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.BCEWithLogitsLoss()

    init_loss = None
    for _ in range(200):
        opt.zero_grad()
        loss = crit(model(X).squeeze(), y)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < init_loss * 0.5, (
        f"Loss didn't decrease enough on {device}: "
        f"{init_loss:.4f} → {loss.item():.4f}"
    )


def test_mlp_regression(device):
    """Train a small MLP on synthetic regression data."""
    n, d = 200, 10
    X = torch.randn(n, d, device=device)
    y = X @ torch.randn(d, 1, device=device) + 0.1 * torch.randn(n, 1, device=device)

    model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = LUMA(model.parameters(), lr=1e-3, weight_decay=0.01)

    init_loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < init_loss * 0.1, (
        f"Loss didn't decrease enough on {device}: "
        f"{init_loss:.4f} → {loss.item():.4f}"
    )


def test_multi_param_groups(device):
    """Different param groups with different LRs should all converge."""
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = LUMA(
        [
            {"params": model[0].parameters(), "lr": 2e-3},
            {"params": model[2].parameters(), "lr": 5e-4},
        ],
        weight_decay=0.01,
    )

    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)

    init_loss = None
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < init_loss * 0.5, (
        f"Loss didn't decrease enough on {device}: "
        f"{init_loss:.4f} → {loss.item():.4f}"
    )
