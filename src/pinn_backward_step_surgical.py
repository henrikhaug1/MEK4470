"""
pinn_backward_step_surgical.py

Backward-facing-step PINN utilities.

This file intentionally stays close to the earlier working backward-step
formulation, but fixes the specific issues that were physically problematic:

  1. The inlet profile is now exactly the intended parabolic profile on the inlet.
  2. The correction factor no longer vanishes along a nonphysical interior curve
     downstream of the step.
  3. The vertical step face is exactly no-slip.
  4. Helper utilities are added for:
       - outlet velocity-gradient diagnostics/losses,
       - a near-wall shear proxy for reattachment estimation.

The training API remains parameter-pytree based:
    params = (ws, bs)
which matches the closest-to-working training script this file is intended to
support.
"""

import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


# =========================================================
# shared parameter helpers
# =========================================================


def pack_params(model):
    ws = tuple(jnp.asarray(w.value) for w in model.ws)
    bs = tuple(jnp.asarray(b.value) for b in model.bs)
    return (ws, bs)


def forward_params(params, xy, activation):
    ws, bs = params
    z = xy
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def net_single(params, activation, xy_single):
    return forward_params(params, xy_single[None], activation)[0]


# =========================================================
# shared optimizer steps
# =========================================================


@functools.partial(jax.jit, static_argnames=("optimizer", "loss_fn"))
def train_step_adam_generic(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_step_lbfgs(params, opt_state, optimizer, value_and_grad_fn):
    value, grad = value_and_grad_fn(params)
    updates, new_state = optimizer.update(
        grad,
        opt_state,
        params=params,
        value=value,
        grad=grad,
        value_fn=lambda p: value_and_grad_fn(p)[0],
    )
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, value


# =========================================================
# backward-facing step — model
# =========================================================


class BackwardStepPINN(nnx.Module):
    def __init__(self, widths, *, key, activation=jax.nn.tanh, n_inputs=2):
        self.activation = activation
        dims = [n_inputs] + list(widths) + [3]  # (x, y) -> (u, v, p)
        keys = jax.random.split(key, len(dims))
        self.ws = nnx.List()
        self.bs = nnx.List()
        for k, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, _ = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / (din + dout))
            b = jnp.zeros((dout,))
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, xy):
        z = xy
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = self.activation(z @ W + b)
        return z @ self.ws[-1] + self.bs[-1]


# =========================================================
# backward-facing step — normalization
# =========================================================


def normalize_xy(x, y, x_min, x_max, H):
    x_n = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y_n = 2.0 * y / H - 1.0
    return x_n, y_n


# =========================================================
# backward-facing step — hard BC ansatz
# =========================================================


def inlet_profile_bs(y, h_step, h_chan, u_mean):
    """Mean-normalised parabolic inlet profile on y ∈ [h_step, h_chan]."""
    return u_mean * 6.0 * (y - h_step) * (h_chan - y) / (h_chan - h_step) ** 2


def _smoothstep01(s):
    """C¹ smoothstep for s already clipped to [0, 1]."""
    return s * s * (3.0 - 2.0 * s)


def _smooth_ramp(z):
    """0 for z≤0, 1 for z≥1, C¹ transition in between."""
    return _smoothstep01(jnp.clip(z, 0.0, 1.0))


def hard_bc_ansatz_bs(
    x,
    y,
    u_raw,
    v_raw,
    p_raw,
    x_min,
    x_max,
    h_step,
    h_chan,
    u_mean,
):
    """
    Hard-BC ansatz for the backward-facing step.

    This keeps the original useful structure

        u = inlet_background + correction * u_raw
        v =                    correction * v_raw
        p = p_raw

    but corrects the two problematic parts of the earlier ansatz:
      - no artificial zero-correction curve inside the downstream fluid,
      - no inlet-profile leakage onto the vertical step face.

    Exactly enforced velocity boundary conditions:
      - inlet x=x_min, y∈[h_step,h_chan]:
            u = parabolic profile, v = 0
      - top wall y=h_chan:
            u = v = 0
      - upstream floor y=h_step, x≤0:
            u = v = 0
      - vertical step face x=0, y≤h_step:
            u = v = 0
      - downstream bottom wall y=0, x≥0:
            u = v = 0

    Pressure remains free:
        p = p_raw
    Outlet pressure and velocity-gradient conditions are handled softly
    in the training loss.
    """
    del x_max  # kept for API compatibility

    k = 15.0

    # ---------------------------------------------------------
    # Inlet background
    # ---------------------------------------------------------
    # Smoothly decays only near the step, preserving the useful original
    # behaviour while being exactly normalised at the inlet.
    alpha = 0.5 * (1.0 + jnp.tanh(k * x))
    alpha_inlet = 0.5 * (1.0 + jnp.tanh(k * x_min))
    inlet_weight = (1.0 - alpha) / (1.0 - alpha_inlet)

    # The inlet profile is defined only on the physical inlet branch y>=h_step.
    # This makes the step face exactly no-slip for y<=h_step.
    u_in = jnp.where(
        y >= h_step,
        inlet_profile_bs(y, h_step, h_chan, u_mean),
        0.0,
    )

    # ---------------------------------------------------------
    # Correction factor
    # ---------------------------------------------------------
    # Zero exactly at the inlet, so v=0 and u equals the inlet profile there.
    x_in = jnp.tanh(k * (x - x_min))

    # Local smoothing widths around the corner. These affect only the geometry
    # blending of the correction factor, not the physical boundary values.
    dx = 0.15 * h_step
    dy = 0.15 * h_step

    downstream_ramp = _smooth_ramp(x / dx)  # 0 for x<=0
    upstream_ramp = _smooth_ramp((-x) / dx)  # 0 for x>=0
    above_step_ramp = _smooth_ramp((y - h_step) / dy)  # 0 for y<=h_step

    # These distance-like factors vanish only on the intended physical
    # boundary segment and do not create a spurious interior zero curve.
    #
    # bottom_distance:
    #   - downstream x>=0: exactly y, hence zero only at y=0
    #   - upstream x<0: positive in the physical fluid region y>=h_step
    bottom_distance = jnp.where(
        x >= 0.0,
        y,
        jnp.sqrt(y**2 + (h_step * upstream_ramp) ** 2),
    )

    # floor_distance:
    #   - upstream x<=0: exactly y-h_step, hence zero only at y=h_step
    #   - downstream x>0: strictly positive in the open channel
    floor_distance = jnp.where(
        x <= 0.0,
        y - h_step,
        jnp.sqrt((y - h_step) ** 2 + (h_step * downstream_ramp) ** 2),
    )

    # step_distance:
    #   - y<=h_step: exactly x, hence zero on the vertical step face x=0
    #   - y>h_step: positive at x=0, so the open upper channel is not clamped
    step_distance = jnp.where(
        y <= h_step,
        x,
        jnp.sqrt(x**2 + (dx * above_step_ramp) ** 2),
    )

    top_distance = h_chan - y

    correction = x_in * top_distance * bottom_distance * floor_distance * step_distance

    u = inlet_weight * u_in + correction * u_raw
    v = correction * v_raw
    p = p_raw

    return u, v, p


# =========================================================
# backward-facing step — physical field helper
# =========================================================


def _net_with_bc_bs(
    params,
    activation,
    z,
    x_min,
    x_max,
    H,
    h_step,
    u_mean,
):
    x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
    raw = net_single(params, activation, jnp.stack([x_n, y_n]))
    u, v, p = hard_bc_ansatz_bs(
        z[0],
        z[1],
        raw[0],
        raw[1],
        raw[2],
        x_min,
        x_max,
        h_step,
        H,
        u_mean,
    )
    return jnp.array([u, v, p])


# =========================================================
# backward-facing step — physics residuals
# =========================================================


def residuals_batch_bs(
    params,
    activation,
    Re,
    x,
    y,
    x_min,
    x_max,
    H,
    h_step=1.0,
    u_mean=1.0,
):
    """
    Steady incompressible Navier–Stokes residuals:
        continuity, x-momentum, y-momentum.
    """
    nu = 1.0 / Re
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def residual_single(xy):
        def net_with_bc(z):
            return _net_with_bc_bs(
                params,
                activation,
                z,
                x_min,
                x_max,
                H,
                h_step,
                u_mean,
            )

        out, d_x = jax.jvp(net_with_bc, (xy,), (ex,))
        _, d_y = jax.jvp(net_with_bc, (xy,), (ey,))

        u, v, _p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]

        _, d_xx = jax.jvp(
            lambda z: jax.jvp(net_with_bc, (z,), (ex,))[1],
            (xy,),
            (ex,),
        )
        _, d_yy = jax.jvp(
            lambda z: jax.jvp(net_with_bc, (z,), (ey,))[1],
            (xy,),
            (ey,),
        )
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        continuity = ux + vy
        mom_x = u * ux + v * uy + px - nu * (u_xx + u_yy)
        mom_y = u * vx + v * vy + py - nu * (v_xx + v_yy)
        return continuity, mom_x, mom_y

    return jax.vmap(residual_single)(xy_batch)


# =========================================================
# backward-facing step — evaluation
# =========================================================


def eval_uvp_batch_bs(
    params,
    activation,
    x,
    y,
    x_min,
    x_max,
    H,
    h_step=1.0,
    u_mean=1.0,
):
    """Evaluate (u, v, p) with hard velocity BCs applied."""

    def eval_single(xi, yi):
        out = _net_with_bc_bs(
            params,
            activation,
            jnp.stack([xi, yi]),
            x_min,
            x_max,
            H,
            h_step,
            u_mean,
        )
        return out[0], out[1], out[2]

    return jax.vmap(eval_single)(x, y)


# =========================================================
# backward-facing step — outlet velocity gradients
# =========================================================


def outlet_velocity_gradients_bs(
    params,
    activation,
    x,
    y,
    x_min,
    x_max,
    H,
    h_step=1.0,
    u_mean=1.0,
):
    """
    Return (u_x, v_x) at requested points.

    Used to softly approximate a zero-normal-gradient outflow condition:
        u_x ≈ 0, v_x ≈ 0
    at x=x_max.
    """
    ex = jnp.array([1.0, 0.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def grad_single(xy):
        def net_with_bc(z):
            return _net_with_bc_bs(
                params,
                activation,
                z,
                x_min,
                x_max,
                H,
                h_step,
                u_mean,
            )

        _, d_x = jax.jvp(net_with_bc, (xy,), (ex,))
        return d_x[0], d_x[1]

    return jax.vmap(grad_single)(xy_batch)


# =========================================================
# backward-facing step — near-wall shear proxy
# =========================================================


def bottom_wall_shear_proxy_bs(
    params,
    activation,
    x,
    x_min,
    x_max,
    H,
    h_step=1.0,
    u_mean=1.0,
    y_eps=1e-3,
):
    """
    Near-wall proxy for ∂u/∂y at the downstream bottom wall:
        ∂u/∂y|_{y=0} ≈ u(x, y_eps) / y_eps

    A small positive y_eps avoids differentiating exactly through the
    piecewise distance-like hard-BC factor at the boundary while preserving
    the physically relevant sign of the wall shear.
    """
    y = jnp.full_like(x, y_eps)
    u_eps, _, _ = eval_uvp_batch_bs(
        params,
        activation,
        x,
        y,
        x_min,
        x_max,
        H,
        h_step,
        u_mean,
    )
    return u_eps / y_eps


# =========================================================
# backward-facing step — samplers
# =========================================================


def sample_interior_bs(key, N, x_min, x_max, h, H):
    N_corner = int(0.5 * N)
    N_rest = N - N_corner
    k1, k2 = jax.random.split(key)

    pts1 = jax.random.uniform(k1, (N_corner, 2))
    x1 = pts1[:, 0] * 4.0
    y1 = pts1[:, 1] * 1.5

    pts2 = jax.random.uniform(k2, (N_rest, 2))
    x2 = pts2[:, 0] * (x_max - x_min) + x_min
    y2 = pts2[:, 1] * H
    mask = ~((x2 < 0.0) & (y2 < h))

    return jnp.concatenate([x1, x2[mask]]), jnp.concatenate([y1, y2[mask]])


def sample_walls_bs(key, N, x_min, x_max, h, H):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = N // 4

    x1 = jax.random.uniform(k1, (n,)) * x_max
    y1 = jnp.zeros(n)

    x2 = jnp.zeros(n)
    y2 = jax.random.uniform(k2, (n,)) * h

    x3 = jax.random.uniform(k3, (n,)) * (x_max - x_min) + x_min
    y3 = jnp.full(n, H)

    x4 = jax.random.uniform(k4, (n,)) * abs(x_min) + x_min
    y4 = jnp.full(n, h)

    return jnp.concatenate([x1, x2, x3, x4]), jnp.concatenate([y1, y2, y3, y4])


def sample_inlet_bs(key, N, x_min, h, H):
    y = jax.random.uniform(key, (N,)) * (H - h) + h
    x = jnp.full(N, x_min)
    return x, y


def sample_outlet_bs(key, N, x_max, H):
    y = jax.random.uniform(key, (N,)) * H
    x = jnp.full(N, x_max)
    return x, y
