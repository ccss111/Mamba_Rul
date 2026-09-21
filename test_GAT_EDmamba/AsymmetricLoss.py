import torch
from torch import nn


class AdaptiveAsymmetricMSELoss(nn.Module):
    """
    Cost-sensitive asymmetric loss for RUL prediction.

    The squared error is computed in the same normalized RUL space as the
    existing MSE training loss. Stage masks are computed after restoring the
    target RUL to the original scale with max_rul.
    """

    def __init__(self, sub_dataset, alpha=2.0, gamma=5.0, delta=0.9, focus_threshold=35.0, cap_threshold=125.0, max_rul=125.0):
        super().__init__()
        if delta < 0:
            raise ValueError("delta must be non-negative.")
        self.sub_dataset = sub_dataset
        if sub_dataset == "FD001":
            alpha = 1.5
            gamma = 2.0
        elif sub_dataset == "FD002":
            alpha = 3.0
            gamma = 5.0
        elif sub_dataset == "FD003":
            alpha = 3.5
            gamma = 5.0
        elif sub_dataset == "FD004":
            alpha = 2.0
            gamma = 5.0
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.focus_threshold = float(focus_threshold)
        self.cap_threshold = float(cap_threshold)
        self.max_rul = float(max_rul)

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        over = (error > 0).to(dtype=error.dtype)

        true_rul = y_true * self.max_rul
        focus = (true_rul < self.focus_threshold).to(dtype=error.dtype)
        cap = (true_rul >= self.cap_threshold).to(dtype=error.dtype)

        weight = (1.0 + self.alpha * over)
        weight = weight * (1.0 + self.gamma * focus)
        weight = weight * (1.0 - (1.0 - self.delta) * cap)

        return torch.mean(error.pow(2) * weight)


class PIVENIntervalLoss(nn.Module):
    """Distribution-free prediction-interval loss from PIVEN."""

    def __init__(self, eta=0.95, lam=1.0, sharpness=20.0):
        super().__init__()
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1].")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if sharpness <= 0.0:
            raise ValueError("sharpness must be positive.")
        self.eta = float(eta)
        self.lam = float(lam)
        self.sharpness = float(sharpness)

    def forward(self, lower, upper, y_true):
        lower, upper, y_true = (value.reshape(-1) for value in (lower, upper, y_true))
        if lower.shape != upper.shape or lower.shape != y_true.shape:
            raise ValueError("lower, upper, and y_true must have the same number of elements.")
        width = (upper - lower).clamp_min(0.0)
        mpiw = width.mean()
        # Direct interval violations avoid sigmoid saturation when a sample is
        # far outside the interval and provide a stable gradient to both bounds.
        lower_violation = torch.relu(lower - y_true)
        upper_violation = torch.relu(y_true - upper)
        # Linear hinge keeps a non-vanishing outward gradient until coverage.
        violation = (lower_violation + upper_violation).mean()
        coverage = ((lower <= y_true) & (y_true <= upper)).to(lower.dtype).mean()
        coverage_factor = 1.0 + torch.relu(
            torch.as_tensor(self.eta, device=lower.device, dtype=lower.dtype) - coverage
        )
        return mpiw + self.lam * coverage_factor * violation


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood in normalized RUL space."""

    def forward(self, mean, sigma, y_true):
        mean, sigma, y_true = (value.reshape(-1) for value in (mean, sigma, y_true))
        sigma = sigma.clamp_min(1e-6)
        return 0.5 * ((y_true - mean).pow(2) / sigma.pow(2) + 2.0 * sigma.log()).mean()


class JointRULLoss(nn.Module):
    """Joint asymmetric RUL loss with selectable PIVEN or Gaussian uncertainty loss."""

    def __init__(self, sub_dataset, xi=0.5, interval_eta=0.95, interval_lambda=1.0,
                 uncertainty_loss="piven", **asymmetric_kwargs):
        super().__init__()
        if not 0.0 <= xi <= 1.0:
            raise ValueError("xi must be between 0 and 1.")
        self.xi = float(xi)
        self.uncertainty_loss = str(uncertainty_loss).lower()
        if self.uncertainty_loss not in {"piven", "gaussian"}:
            raise ValueError("uncertainty_loss must be 'piven' or 'gaussian'.")
        self.rul_loss = AdaptiveAsymmetricMSELoss(sub_dataset, **asymmetric_kwargs)
        self.interval_loss = PIVENIntervalLoss(interval_eta, interval_lambda)
        self.gaussian_loss = GaussianNLLLoss()

    def forward(self, point, uncertainty_outputs, y_true):
        loss_rul = self.rul_loss(point, y_true)
        if self.uncertainty_loss == "gaussian":
            loss_pi = self.gaussian_loss(point, uncertainty_outputs["sigma"], y_true)
        else:
            loss_pi = self.interval_loss(
                uncertainty_outputs["lower"], uncertainty_outputs["upper"], y_true
            )
        return self.xi * loss_pi + (1.0 - self.xi) * loss_rul
