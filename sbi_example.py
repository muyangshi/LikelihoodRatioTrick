# %% ── 1. Imports ──────────────────────────────────────────────────────────────────
import torch
from sbi.inference import NRE_A
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cpu"                # use "cuda" for GPU

# %% ── 2. Simulator and prior ──────────────────────────────────────────────────────
def simulator(theta: torch.Tensor) -> torch.Tensor:
    """Simulate a single scalar observation x ~ N(mu, 1) given mu=theta."""
    return theta + torch.randn_like(theta)           # σ = 1 fixed

# Treat μ ∈ [−5, 5] as plausible a‑priori
prior = BoxUniform(low=-8.*torch.ones(1), high=8.*torch.ones(1), device=device)

# %% ── 3. Generate training data ───────────────────────────────────────────────────
num_sim = 100_000
theta_train = prior.sample((num_sim,))               # (N, 1)
x_train     = simulator(theta_train)                 # (N, 1)

# %% ── 4. Train NRE‑A (single round, amortised) ───────────────────────────────────
inference   = NRE_A(prior=prior, device=device)
inference.append_simulations(theta_train, x_train)
classifier  = inference.train()                     # returns the trained network   [oai_citation:0‡sbi-dev.github.io](https://sbi-dev.github.io/sbi/latest/tutorials/18_training_interface/)

# %% ── 5. Build a log‑ratio estimator convenience wrapper ──────────────────────────
# @torch.no_grad()
# def log_r_hat(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """Approximate log [p(x|mu)/p(x)]."""
#     return classifier(mu.to(device), x.to(device)).squeeze(-1)

@torch.no_grad()
def log_r_hat(mu, x):
    if x.shape[0] == 1:                         # single observation → tile it
        x = x.expand(mu.shape[0], -1)
    elif x.shape[0] != mu.shape[0]:
        raise ValueError("theta and x must share the same batch dimension")
    return classifier(mu, x).squeeze(-1)

# %% ── 6. Example: evaluate the likelihood ratio on a grid ─────────────────────────
mu_grid   = torch.linspace(-4, 4, 200).unsqueeze(1)    # (200, 1)
x_obs     = torch.tensor([[0.3]])                      # observed datum
x_obs_rep = x_obs.expand(mu_grid.size(0), -1)

log_r         = log_r_hat(mu_grid, x_obs_rep)                  # (200,)
log_like_true = -0.5*((x_obs - mu_grid)**2).squeeze(-1)        # Normal σ=1, up to constant

# "normalise away the additive constant"
log_r_norm         = log_r - log_r.max()                  # shift so max = 0
log_like_true_norm = log_like_true - log_like_true.max()

# (a) Overlay the two curves
plt.figure()
plt.plot(mu_grid.numpy(), log_like_true_norm.numpy(), label="True log p(x|μ)  (normalised)")
plt.plot(mu_grid.numpy(), log_r_norm.numpy(),    label="Estimated log‑ratio (normalised)")
plt.xlabel("μ")
plt.ylabel("log‑density (shifted)")
plt.title("True vs. estimated log‑density curves")
plt.legend()
plt.tight_layout()
plt.show()

# (b) Residuals (estimate – truth)
residuals = (log_r_norm - log_like_true_norm).numpy()
plt.figure()
plt.plot(mu_grid.numpy(), residuals, label="Residual")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("μ")
plt.ylabel("Residual")
plt.title("Residuals across μ")
plt.legend()
plt.tight_layout()
plt.show()

# (c) GOF
fig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
ax.scatter(log_like_true.numpy(), log_r.numpy())
ax.axline((0, 0), slope=1, color='black', linestyle='--')
ax.set_title(rf'Goodness of Fit')
ax.set_xlabel('True log p(x|μ) (normalised)')
ax.set_ylabel('Estimated log p(x|μ) (normalised)')
plt.tight_layout()
plt.show()

# # Optional: turn log_r into an (unnormalised) posterior by adding log‑prior
# log_post = log_r + prior.log_prob(mu_grid)
# post     = torch.softmax(log_post, dim=0)            # normalised grid posterior