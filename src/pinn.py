import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax


jax.config.update("jax_enable_x64", True)


# =========================================================
# Model save / load (generic for all PINN models)
# =========================================================


def save_model(model, path):
    state = nnx.state(model, nnx.Param)
    flat, _ = jax.tree.flatten(state)
    arrays = {f"p{i}": np.array(a) for i, a in enumerate(flat)}
    n_layers = len(model.ws)
    widths = np.array([int(model.ws[i].value.shape[1]) for i in range(n_layers - 1)])
    n_inputs = np.array(int(model.ws[0].value.shape[0]))
    np.savez(path, _n=np.array(len(flat)), _widths=widths, _n_inputs=n_inputs, **arrays)


def load_model_state(model, path):
    data = np.load(path)
    n = int(data["_n"])
    flat = [jnp.array(data[f"p{i}"]) for i in range(n)]
    state = nnx.state(model, nnx.Param)
    _, treedef = jax.tree.flatten(state)
    new_state = jax.tree.unflatten(treedef, flat)
    nnx.update(model, new_state)


# =========================================================
# L-BFGS helper (nnx.split/merge pattern)
# =========================================================


def lbfgs_step(graphdef, params, rest, opt_state, opt, loss_fn_of_params):
    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_of_params)
    value, grads = value_and_grad_fn(params, state=opt_state)
    updates, opt_state = opt.update(
        grads,
        opt_state,
        params,
        value=value,
        grad=grads,
        value_fn=loss_fn_of_params,
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, value


# =========================================================
# backward-facing step — model
# =========================================================


class BackwardStepPINN(nnx.Module):
    def __init__(self, widths, *, key, activation=jax.nn.tanh, n_inputs=2):
        self.activation = activation
        dims = [n_inputs] + list(widths) + [3]
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
    return u_mean * 6.0 * (y - h_step) * (h_chan - y) / (h_chan - h_step) ** 2


def hard_bc_ansatz_bs(x, y, u_raw, v_raw, p_raw, x_min, x_max, h_step, h_chan, u_mean):
    k = 15.0
    alpha = 0.5 * (1.0 + jnp.tanh(k * x))
    step_blend = 0.5 * (1.0 - jnp.tanh(k * (y - h_step)))
    f_step = jnp.tanh(k * x) * step_blend + (1.0 - step_blend)
    y_lo = h_step * (1.0 - alpha)
    D = (y - y_lo) * (h_chan - y)
    u_in = (
        inlet_profile_bs(y, h_step, h_chan, u_mean)
        * 0.5
        * (1.0 + jnp.tanh(k * (y - h_step)))
    )
    x_in = jnp.tanh(k * (x - x_min))
    correction = x_in * D * f_step
    u = u_in * (1.0 - alpha) + correction * u_raw
    v = correction * v_raw
    p = p_raw
    return u, v, p


# =========================================================
# backward-facing step — physics residuals (hard BCs)
# =========================================================


def residuals_batch_bs(model, Re, x, y, x_min, x_max, H, h_step=1.0, u_mean=1.0):
    nu = 1.0 / Re
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def residual_single(xy):
        def net_with_bc(z):
            x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
            raw = model(jnp.stack([x_n, y_n])[None])[0]
            u, v, p = hard_bc_ansatz_bs(
                z[0], z[1], raw[0], raw[1], raw[2],
                x_min, x_max, h_step, H, u_mean,
            )
            return jnp.array([u, v, p])

        out, d_x = jax.jvp(net_with_bc, (xy,), (ex,))
        _, d_y = jax.jvp(net_with_bc, (xy,), (ey,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]

        _, d_xx = jax.jvp(lambda z: jax.jvp(net_with_bc, (z,), (ex,))[1], (xy,), (ex,))
        _, d_yy = jax.jvp(lambda z: jax.jvp(net_with_bc, (z,), (ey,))[1], (xy,), (ey,))
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        continuity = ux + vy
        mom_x = u * ux + v * uy + px - nu * (u_xx + u_yy)
        mom_y = u * vx + v * vy + py - nu * (v_xx + v_yy)
        return continuity, mom_x, mom_y

    return jax.vmap(residual_single)(xy_batch)


# =========================================================
# backward-facing step — evaluation (hard BCs)
# =========================================================


def eval_uvp_batch_bs(model, x, y, x_min, x_max, H, h_step=1.0, u_mean=1.0):
    def eval_single(xi, yi):
        x_n, y_n = normalize_xy(xi, yi, x_min, x_max, H)
        raw = model(jnp.stack([x_n, y_n])[None])[0]
        return hard_bc_ansatz_bs(
            xi, yi, raw[0], raw[1], raw[2],
            x_min, x_max, h_step, H, u_mean,
        )

    return jax.vmap(eval_single)(x, y)


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


# =========================================================
# Taylor-Green — model
# =========================================================


class TaylorGreenPINN(nnx.Module):
    def __init__(self, widths, *, key, activation=jax.nn.tanh, n_inputs=5):
        self.activation = activation
        dims = [n_inputs] + list(widths) + [3]
        keys = jax.random.split(key, len(dims))
        self.ws = nnx.List()
        self.bs = nnx.List()
        for k, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, _ = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / (din + dout))
            b = jnp.zeros((dout,))
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, xyt):
        z = xyt
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = self.activation(z @ W + b)
        return z @ self.ws[-1] + self.bs[-1]


# =========================================================
# Taylor-Green — physics residuals
# =========================================================


def _net_tg_physical(model, z, t_max):
    x_, y_, t_ = z[0], z[1], z[2]
    t_n = 2.0 * t_ / t_max - 1.0
    inp = jnp.stack([jnp.sin(x_), jnp.cos(x_), jnp.sin(y_), jnp.cos(y_), t_n])
    raw = model(inp[None])[0]
    u_ic = jnp.cos(x_) * jnp.sin(y_)
    v_ic = -jnp.sin(x_) * jnp.cos(y_)
    p_ic = -0.25 * (jnp.cos(2.0 * x_) + jnp.cos(2.0 * y_))
    return jnp.array([u_ic + t_ * raw[0], v_ic + t_ * raw[1], p_ic + t_ * raw[2]])


def residuals_batch(model, Re, x, y, t, t_max):
    nu = 1.0 / Re
    ex = jnp.array([1.0, 0.0, 0.0])
    ey = jnp.array([0.0, 1.0, 0.0])
    et = jnp.array([0.0, 0.0, 1.0])
    xyt_batch = jnp.stack([x, y, t], axis=-1)

    def residual_single(xyt):
        def net_physical(z):
            return _net_tg_physical(model, z, t_max)

        out, d_x = jax.jvp(net_physical, (xyt,), (ex,))
        _, d_y = jax.jvp(net_physical, (xyt,), (ey,))
        _, d_t = jax.jvp(net_physical, (xyt,), (et,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]
        ut, vt = d_t[0], d_t[1]

        _, d_xx = jax.jvp(
            lambda z: jax.jvp(net_physical, (z,), (ex,))[1], (xyt,), (ex,)
        )
        _, d_yy = jax.jvp(
            lambda z: jax.jvp(net_physical, (z,), (ey,))[1], (xyt,), (ey,)
        )
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        continuity = ux + vy
        mom_x = ut + u * ux + v * uy + px - nu * (u_xx + u_yy)
        mom_y = vt + u * vx + v * vy + py - nu * (v_xx + v_yy)
        return continuity, mom_x, mom_y

    return jax.vmap(residual_single)(xyt_batch)


# =========================================================
# Taylor-Green — evaluation
# =========================================================


def eval_uvp_batch(model, x, y, t, t_max=2.0):
    def eval_single(xi, yi, ti):
        out = _net_tg_physical(model, jnp.stack([xi, yi, ti]), t_max)
        return out[0], out[1], out[2]

    return jax.vmap(eval_single)(x, y, t)


# =========================================================
# Taylor-Green — samplers
# =========================================================


def sample_interior(key, N, t_max):
    pts = jax.random.uniform(key, (N, 3))
    x = pts[:, 0] * 2.0 * jnp.pi
    y = pts[:, 1] * 2.0 * jnp.pi
    t = pts[:, 2] * t_max
    return x, y, t


# =========================================================
# Taylor-Green — loss
# =========================================================


def loss_fn_tg(model, x_col, y_col, t_col, Re, t_max=2.0):
    cont, mom_x, mom_y = residuals_batch(model, Re, x_col, y_col, t_col, t_max)
    return jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)


# =========================================================
# lid-driven cavity — hard BC ansatz
# =========================================================


def hard_bc_ansatz_lc(x, y, u_raw, v_raw, p_raw):
    lid = (1.0 - x**2) ** 2
    u = lid * (1.0 + y) / 2.0 + (1.0 - y**2) * (1.0 - x**2) * u_raw
    v = (1.0 - y**2) * (1.0 - x**2) * v_raw
    return u, v, p_raw


# =========================================================
# lid-driven cavity — physics residuals
# =========================================================


def residuals_batch_lc(model, Re, x, y):
    nu = 2.0 / Re
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def residual_single(xy):
        def net_with_bc(z):
            raw = model(z[None])[0]
            u, v, p = hard_bc_ansatz_lc(z[0], z[1], raw[0], raw[1], raw[2])
            return jnp.array([u, v, p])

        out, d_x = jax.jvp(net_with_bc, (xy,), (ex,))
        _, d_y = jax.jvp(net_with_bc, (xy,), (ey,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]

        _, d_xx = jax.jvp(lambda z: jax.jvp(net_with_bc, (z,), (ex,))[1], (xy,), (ex,))
        _, d_yy = jax.jvp(lambda z: jax.jvp(net_with_bc, (z,), (ey,))[1], (xy,), (ey,))
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        continuity = ux + vy
        mom_x = u * ux + v * uy + px - nu * (u_xx + u_yy)
        mom_y = u * vx + v * vy + py - nu * (v_xx + v_yy)
        return continuity, mom_x, mom_y

    return jax.vmap(residual_single)(xy_batch)


# =========================================================
# lid-driven cavity — evaluation
# =========================================================


def eval_uvp_batch_lc(model, x, y):
    def eval_single(xi, yi):
        raw = model(jnp.stack([xi, yi])[None])[0]
        return hard_bc_ansatz_lc(xi, yi, raw[0], raw[1], raw[2])

    return jax.vmap(eval_single)(x, y)


# =========================================================
# lid-driven cavity — sampler
# =========================================================


def sample_interior_lc(key, N):
    pts = jax.random.uniform(key, (N, 2)) * 2.0 - 1.0
    return pts[:, 0], pts[:, 1]
