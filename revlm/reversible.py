"""Reversible residual stacks: activations are rebuilt in the backward pass, not stored.

A normal residual stack x_{n+1} = x_n + f_n(x_n) is a forward-Euler integrator. Autograd
must keep every x_n (and every intermediate inside f_n) alive until backward reaches that
layer, so activation memory grows linearly with depth.

If instead each layer is an *invertible* map of a small state, backward can walk the stack
from the top, reconstructing layer n's input from its output, recomputing that one layer
with grad enabled, and backpropagating through it before moving down. Only the final
state is ever stored, so activation memory is O(1) in depth. The price is one extra
forward pass per layer (~4/3 of normal training compute).

Every variant below is a map on a tuple `state`, with:

    init(x)            embedding output -> initial state
    step(blk, s)       one layer forward                    (used by forward, no grad)
    back(blk, s, g)    given the layer's OUTPUT state and the grads w.r.t. it, return the
                       layer's INPUT state and the grads w.r.t. it; parameter grads are
                       accumulated into .grad as a side effect
    readout(s)         final state -> hidden vector fed to the LM head

Each `back` recomputes the layer's sublayers exactly once with grad enabled and reuses
that single recompute both to invert the layer and to backpropagate through it.

`blk.F` is the attention sublayer, `blk.G` the MLP sublayer and `blk.f(x) = F(x) +
G(x + F(x))`, the whole pre-LN block written as a residual update (x + blk.f(x) is
exactly an ordinary transformer block).
"""
from __future__ import annotations

import torch
from torch.amp import custom_bwd, custom_fwd


def _vjp(out: torch.Tensor, grad: torch.Tensor) -> None:
    # Autocast makes sublayer outputs bf16/fp16 while the residual state is fp32. Ordinary
    # autograd casts the incoming fp32 grad down to the branch dtype at the add; do the same.
    torch.autograd.backward(out, grad.to(out.dtype))


def _leaf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().requires_grad_(True)


class RevNet:
    """Additive coupling (Gomez et al. 2017; Reformer). Two streams, exactly invertible:

        a' = a + F(b)            b  = b' - G(a')
        b' = b + G(a')     <=>   a  = a' - F(b)

    Numerically this is symplectic Euler on a split state, which is why it is also called
    the "Hamiltonian" or "leapfrog-coupled" reversible block. Readout averages the streams.
    """

    name, exact = "revnet", True

    def init(self, x):
        return (x, x)

    def step(self, blk, s):
        a, b = s
        a = a + blk.F(b)
        b = b + blk.G(a)
        return (a, b)

    def readout(self, s):
        return 0.5 * (s[0] + s[1])

    def back(self, blk, s, g):
        a1, b1 = s
        ga1, gb1 = g
        with torch.enable_grad():
            a1_ = _leaf(a1)
            go = blk.G(a1_)
            _vjp(go, gb1)
        with torch.no_grad():
            b0 = b1 - go
            ga = ga1 + a1_.grad  # total grad w.r.t. a' == grad w.r.t. a
        with torch.enable_grad():
            b0_ = _leaf(b0)
            fo = blk.F(b0_)
            _vjp(fo, ga)
        with torch.no_grad():
            a0 = a1 - fo
            gb = gb1 + b0_.grad
        return (a0, b0), (ga, gb)


class Midpoint:
    """Explicit midpoint / two-step leapfrog (Chang et al. 2018, "Midpoint network"):

        x_{n+1} = x_{n-1} + 2h f(x_n),   h = 1/2   =>   x_{n-1} = x_{n+1} - 2h f(x_n)

    State is the pair (x_{n-1}, x_n). With h = 1/2 each update has the same magnitude as an
    ordinary residual block. Known weakness: even and odd layers only talk to each other
    through f, and the scheme has a parasitic oscillating mode.
    """

    name, exact = "midpoint", True

    def init(self, x):
        return (x, x)

    def step(self, blk, s):
        u, v = s
        return (v, u + blk.f(v))

    def readout(self, s):
        return s[1]

    def back(self, blk, s, g):
        p, q = s  # p = x_n, q = x_{n+1}
        gp, gq = g
        with torch.enable_grad():
            p_ = _leaf(p)
            fo = blk.f(p_)
            _vjp(fo, gq)
        with torch.no_grad():
            u = q - fo
            gv = gp + p_.grad
        return (u, p), (gq, gv)


class Leapfrog:
    """Second-order leapfrog (Chang et al. 2018, "Leapfrog network"), h = 1:

        x_{n+1} = 2 x_n - x_{n-1} + h^2 f(x_n)   =>   x_{n-1} = 2 x_n - x_{n+1} + h^2 f(x_n)

    f acts as an acceleration: the difference x_n - x_{n-1} is a velocity that keeps
    accumulating, so the residual stream can grow quadratically with depth.
    """

    name, exact = "leapfrog", True

    def init(self, x):
        return (x, x)

    def step(self, blk, s):
        u, v = s
        return (v, 2 * v - u + blk.f(v))

    def readout(self, s):
        return s[1]

    def back(self, blk, s, g):
        p, q = s
        gp, gq = g
        with torch.enable_grad():
            p_ = _leaf(p)
            fo = blk.f(p_)
            _vjp(fo, gq)
        with torch.no_grad():
            u = 2 * p - q + fo
            gv = gp + 2 * gq + p_.grad
        return (u, p), (-gq, gv)


class Euler:
    """The ordinary residual stack, x_{n+1} = x_n + f(x_n), made "reversible" by solving

        x_n = x_{n+1} - f(x_n)

    with `iters` fixed-point iterations. This is NOT exact: it converges only where f is a
    contraction, and costs `iters` extra forwards per layer. Its forward pass is
    bit-identical to the standard model; only the gradients are approximate. The largest
    relative inversion residual |z + f(z) - x_{n+1}| / |x_{n+1}| seen since the last read is
    kept in `max_resid`.
    """

    name, exact = "euler", False

    def __init__(self, iters: int = 6):
        self.iters = iters
        self.max_resid = None

    def init(self, x):
        return (x,)

    def step(self, blk, s):
        return (s[0] + blk.f(s[0]),)

    def readout(self, s):
        return s[0]

    def back(self, blk, s, g):
        (y,), (gy,) = s, g
        with torch.no_grad():
            z = y
            for _ in range(self.iters):
                z = y - blk.f(z)
        with torch.enable_grad():
            z_ = _leaf(z)
            fo = blk.f(z_)
            _vjp(fo, gy)
        with torch.no_grad():
            r = (z + fo - y).norm() / y.norm().clamp_min(1e-12)
            self.max_resid = r if self.max_resid is None else torch.maximum(self.max_resid, r)
        return (z,), (gy + z_.grad,)

    def pop_resid(self) -> float | None:
        r, self.max_resid = self.max_resid, None
        return None if r is None else float(r)


def make_variant(mode: str, euler_iters: int = 6):
    if mode == "revnet":
        return RevNet()
    if mode == "midpoint":
        return Midpoint()
    if mode == "leapfrog":
        return Leapfrog()
    if mode in ("euler", "standard", "ckpt"):
        return Euler(euler_iters)
    raise ValueError(mode)


class RevStack(torch.autograd.Function):
    """Runs `blocks` under no_grad and saves only the final state; backward reconstructs.

    Parameters are deliberately not inputs to the Function: their grads are accumulated
    by the per-layer backward calls inside `back`, exactly as reentrant checkpointing does.
    custom_fwd/custom_bwd make the backward recompute run under the same autocast state
    as the forward, so the recomputed f is the same f.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, blocks, variant, *state):
        inputs = state
        with torch.no_grad():
            for blk in blocks:
                state = variant.step(blk, state)
        # an output that is literally an input tensor (possible with 1 layer) confuses
        # autograd's bookkeeping; hand back a copy instead
        state = tuple(s.clone() if any(s is i for i in inputs) else s for s in state)
        ctx.blocks, ctx.variant = blocks, variant
        ctx.save_for_backward(*state)
        return state

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        state = tuple(t.detach() for t in ctx.saved_tensors)
        for blk in reversed(ctx.blocks):
            state, grads = ctx.variant.back(blk, state, grads)
        return (None, None, *grads)
