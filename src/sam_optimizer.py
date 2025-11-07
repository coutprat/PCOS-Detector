import torch
from typing import Iterable, Optional
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    Use this wrapper with your base optimizer.
    Example:
        base_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        optimizer = SAM(model.parameters(), base_opt, rho=0.05, adaptive=True)

        # training step:
        # 1) loss.backward()
        #    optimizer.first_step()
        # 2) second forward/backward
        #    optimizer.second_step()
    """

    def __init__(self, params: Iterable, base_optimizer: Optimizer,
                 rho: float = 0.05, adaptive: bool = True):
        if rho <= 0:
            raise ValueError("rho must be > 0")
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute L2 norm of all gradients."""
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = p.abs() if adaptive else 1.0
                norms.append((scale * p.grad).norm(p=2).to(device))
        if len(norms) == 0:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        """Perturb weights toward gradient ascent direction."""
        scale = self.param_groups[0]["rho"] / (self._grad_norm() + 1e-12)
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p.abs() if adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        """Return to original weights and perform optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, *args, **kwargs):
        raise RuntimeError("Call first_step(), then second forward/backward, then second_step().")

    def zero_grad(self, set_to_none: Optional[bool] = None):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
