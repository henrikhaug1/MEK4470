import jax
import jax.numpy as jnp


# =========================================================
# normalization
# =========================================================


def normalize_xy(x, y, x_min, x_max, H):
    x_n = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y_n = 2.0 * y / H - 1.0
    return x_n, y_n


# =========================================================
# hard BC ansatz
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
# physics residuals (hard BCs)
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
# evaluation (hard BCs)
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
# raw output (no ansatz, for soft-BC training)
# =========================================================


def eval_uvp_batch_bs_raw(model, x, y, x_min, x_max, H):
    def eval_single(xi, yi):
        x_n, y_n = normalize_xy(xi, yi, x_min, x_max, H)
        raw = model(jnp.stack([x_n, y_n])[None])[0]
        return raw[0], raw[1], raw[2]

    return jax.vmap(eval_single)(x, y)


def dudx_at_outlet_bs_raw(model, x, y, x_min, x_max, H):
    ex = jnp.array([1.0, 0.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def single(xy):
        def net_fn(z):
            x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
            return model(jnp.stack([x_n, y_n])[None])[0]

        _, d_x = jax.jvp(net_fn, (xy,), (ex,))
        return d_x[0], d_x[1]

    return jax.vmap(single)(xy_batch)


def residuals_batch_bs_raw(model, Re, x, y, x_min, x_max, H):
    nu = 1.0 / Re
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def residual_single(xy):
        def net_fn(z):
            x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
            return model(jnp.stack([x_n, y_n])[None])[0]

        out, d_x = jax.jvp(net_fn, (xy,), (ex,))
        _, d_y = jax.jvp(net_fn, (xy,), (ey,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]

        _, d_xx = jax.jvp(lambda z: jax.jvp(net_fn, (z,), (ex,))[1], (xy,), (ex,))
        _, d_yy = jax.jvp(lambda z: jax.jvp(net_fn, (z,), (ey,))[1], (xy,), (ey,))
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        continuity = ux + vy
        mom_x = u * ux + v * uy + px - nu * (u_xx + u_yy)
        mom_y = u * vx + v * vy + py - nu * (v_xx + v_yy)
        return continuity, mom_x, mom_y

    return jax.vmap(residual_single)(xy_batch)


def residual_parts_bs_raw(model, x, y, x_min, x_max, H):
    """Split the momentum residual into its Re-independent and viscous parts.

    Returns (cont, a_x, b_x, a_y, b_y) where, per momentum component,
        mom = a - (1/Re) * b
    with
        a = u·∇u + ∇p   (inertial + pressure, independent of Re)
        b = ∇²u         (Laplacian; the viscous term is (1/Re)·b)

    This lets the optimal viscosity be read off in closed form as a
    least-squares projection  ν* = Σ(a·b)/Σ(b·b)  instead of being a free
    parameter that can run away by suppressing the viscous term.
    """
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def single(xy):
        def net_fn(z):
            x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
            return model(jnp.stack([x_n, y_n])[None])[0]

        out, d_x = jax.jvp(net_fn, (xy,), (ex,))
        _, d_y = jax.jvp(net_fn, (xy,), (ey,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]

        _, d_xx = jax.jvp(lambda z: jax.jvp(net_fn, (z,), (ex,))[1], (xy,), (ex,))
        _, d_yy = jax.jvp(lambda z: jax.jvp(net_fn, (z,), (ey,))[1], (xy,), (ey,))
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        cont = ux + vy
        a_x = u * ux + v * uy + px
        b_x = u_xx + u_yy
        a_y = u * vx + v * vy + py
        b_y = v_xx + v_yy
        return cont, a_x, b_x, a_y, b_y

    return jax.vmap(single)(xy_batch)


# =========================================================
# samplers
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


def dudy_batch_bs_raw(model, x, y, x_min, x_max, H):
    """Compute ∂u/∂y at each (x, y) point using JAX forward-mode autodiff."""
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def single(xy):
        def net_fn(z):
            x_n, y_n = normalize_xy(z[0], z[1], x_min, x_max, H)
            return model(jnp.stack([x_n, y_n])[None])[0]

        _, d_y = jax.jvp(net_fn, (xy,), (ey,))
        return d_y[0]  # ∂u/∂y

    return jax.vmap(single)(xy_batch)
