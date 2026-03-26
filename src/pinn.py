import jax

jax.config.update("jax_enable_x64", True)

import flax.nnx as nnx
import jax.numpy as jnp
import optax
import functools

from src.analytical import taylor_green_p, taylor_green_u, taylor_green_v


class NavierStokesPINN(nnx.Module):
    def __init__(self, widths: list, *, key, Re, activation=jax.nn.tanh):
        self.Re = Re
        keys = jax.random.split(key, len(widths) + 1)
        dims = [3] + list(widths) + [3]  # input: (x, y, t), output: (u, v, p)
        self.ws = nnx.List()
        self.bs = nnx.List()
        self.activation = activation
        for k, din, dout in zip(
            keys, dims[:-1], dims[1:]
        ):  # zips pairs in lists [3, x_1, x_2] [x_1, x_2, 3]
            wk, bk = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / din)
            b = jnp.zeros((dout,))
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, xyt):
        z = xyt
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = self.activation(z @ W + b)
        out = z @ self.ws[-1] + self.bs[-1]
        u, v, p = out[:, 0], out[:, 1], out[:, 2]
        return u, v, p


def pack_params(model):
    ws = tuple(jnp.asarray(W.value) for W in model.ws)
    bs = tuple(jnp.asarray(b.value) for b in model.bs)
    return (ws, bs)


def apply_params(model, params):
    ws, bs = params
    for Wi, bi, Wp, bp in zip(model.ws, model.bs, ws, bs):
        Wi.value = Wp
        bi.value = bp


def forward_params(params, xyt2d, activation):
    ws, bs = params
    z = xyt2d
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def net_single(params, activation, xyt_single):
    """Evaluate network at a single point xyt_single = [x, y, t], returns [u, v, p]."""
    return forward_params(params, xyt_single[None], activation)[0]


def residuals_batch(params, activation, Re, x_col, y_col, t_col):
    xyt = jnp.stack([x_col, y_col, t_col], axis=-1)  # (N, 3)

    e_x = jnp.array([1.0, 0.0, 0.0])
    e_y = jnp.array([0.0, 1.0, 0.0])
    e_t = jnp.array([0.0, 0.0, 1.0])

    def net(xyt_single):
        return net_single(params, activation, xyt_single)

    def residual_single(xyt_single):
        (uvp, d_x), (_, d2_xx) = jax.jvp(
            lambda xyt: jax.jvp(net, (xyt,), (e_x,)),
            (xyt_single,),
            (e_x,),
        )
        u, v = uvp[0], uvp[1]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uxx, vxx = d2_xx[0], d2_xx[1]
        (_, d_y), (_, d2_yy) = jax.jvp(
            lambda xyt: jax.jvp(net, (xyt,), (e_y,)),
            (xyt_single,),
            (e_y,),
        )
        uy, vy, py = d_y[0], d_y[1], d_y[2]
        uyy, vyy = d2_yy[0], d2_yy[1]

        # JVP in t: du/dt, dv/dt — cost: ~1 network eval
        _, d_t = jax.jvp(net, (xyt_single,), (e_t,))
        ut, vt = d_t[0], d_t[1]

        mom_x = ut + u * ux + v * uy + px - (1 / Re) * (uxx + uyy)
        mom_y = vt + u * vx + v * vy + py - (1 / Re) * (vxx + vyy)
        cont = ux + vy
        return mom_x, mom_y, cont

    return jax.vmap(residual_single)(xyt)


def eval_uvp_batch(params, activation, x, y, t):
    """Evaluate u, v, p for a batch of points in a single forward pass."""
    xyt = jnp.stack([x, y, t], axis=-1)  # (N, 3)
    out = forward_params(params, xyt, activation)  # (N, 3)
    return out[:, 0], out[:, 1], out[:, 2]


# ---------- Taylor green sampling ----------
def sample_interior(key, N, t_max):
    pts = jax.random.uniform(key, (N, 3))
    x = pts[:, 0] * 2 * jnp.pi
    y = pts[:, 1] * 2 * jnp.pi
    t = pts[:, 2] * t_max
    return x, y, t


def sample_ic(key, N):
    pts = jax.random.uniform(key, (N, 2))
    x = pts[:, 0] * 2 * jnp.pi
    y = pts[:, 1] * 2 * jnp.pi
    t = jnp.zeros(N)
    return x, y, t


# Loss for Taylor Green
def loss_fn(
    params,
    activation,
    x_col,
    y_col,
    t_col,
    x_ic,
    y_ic,
    t_ic,
    x_data,
    y_data,
    t_data,
    Re,
):
    # Physical residual — one jacfwd sweep over all collocation points
    mom_x, mom_y, cont = residuals_batch(params, activation, Re, x_col, y_col, t_col)
    physics_loss = jnp.mean(mom_x**2) + jnp.mean(mom_y**2) + jnp.mean(cont**2)

    # IC loss — single batched forward pass
    u_ic, v_ic, _ = eval_uvp_batch(params, activation, x_ic, y_ic, t_ic)
    u_ic_ref = taylor_green_u(x_ic, y_ic, 0.0, 1 / Re)
    v_ic_ref = taylor_green_v(x_ic, y_ic, 0.0, 1 / Re)
    ic_loss = jnp.mean((u_ic - u_ic_ref) ** 2) + jnp.mean((v_ic - v_ic_ref) ** 2)

    # Data loss — single batched forward pass
    u_data, v_data, p_data = eval_uvp_batch(params, activation, x_data, y_data, t_data)
    u_ref = taylor_green_u(x_data, y_data, t_data, 1 / Re)
    v_ref = taylor_green_v(x_data, y_data, t_data, 1 / Re)
    p_ref = taylor_green_p(x_data, y_data, t_data, 1 / Re)
    data_loss = (
        jnp.mean((u_data - u_ref) ** 2)
        + jnp.mean((v_data - v_ref) ** 2)
        + jnp.mean((p_data - p_ref) ** 2)
    )
    return physics_loss + 10.0 * ic_loss + 10.0 * data_loss


def make_loss_for_lbfgs(
    activation, x_col, y_col, t_col, x_ic, y_ic, t_ic, x_data, y_data, t_data, Re
):
    """Return a loss function that takes only params (for L-BFGS)."""

    def fn(params):
        return loss_fn(
            params,
            activation,
            x_col,
            y_col,
            t_col,
            x_ic,
            y_ic,
            t_ic,
            x_data,
            y_data,
            t_data,
            Re,
        )

    return fn


# Loss for Backward step
def loss_fn_backward_step(): ...


def train_step_lbfgs(params, opt_state, optimizer, value_and_grad_fn):
    """Single L-BFGS step with line search."""
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


@functools.partial(jax.jit, static_argnames=("activation", "optimizer"))
def train_step_adam(
    params,
    opt_state,
    optimizer,
    activation,
    x_col,
    y_col,
    t_col,
    x_ic,
    y_ic,
    t_ic,
    x_data,
    y_data,
    t_data,
    Re,
):
    loss, grads = jax.value_and_grad(loss_fn)(
        params,
        activation,
        x_col,
        y_col,
        t_col,
        x_ic,
        y_ic,
        t_ic,
        x_data,
        y_data,
        t_data,
        Re,
    )
    updates, opt_state_new = optimizer.update(grads, opt_state)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss
