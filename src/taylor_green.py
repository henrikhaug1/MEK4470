import jax
import jax.numpy as jnp


# =========================================================
# physics residuals
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


def residual_parts(model, x, y, t, t_max):
    """Split the Taylor-Green momentum residual into Re-independent and viscous parts.

    Returns (cont, a_x, b_x, a_y, b_y) where, per momentum component,
        mom = a - nu * b,   nu = 1/Re,
    with
        a = u_t + u·∇u + ∇p   (unsteady + inertial + pressure, Re-independent)
        b = ∇²u               (Laplacian; viscous term is nu·b)

    The optimal viscosity is then the least-squares projection
        nu* = Σ(a·b)/Σ(b·b)   ->   Re = 1/nu*.
    """
    ex = jnp.array([1.0, 0.0, 0.0])
    ey = jnp.array([0.0, 1.0, 0.0])
    et = jnp.array([0.0, 0.0, 1.0])
    xyt_batch = jnp.stack([x, y, t], axis=-1)

    def single(xyt):
        def net_physical(z):
            return _net_tg_physical(model, z, t_max)

        out, d_x = jax.jvp(net_physical, (xyt,), (ex,))
        _, d_y = jax.jvp(net_physical, (xyt,), (ey,))
        _, d_t = jax.jvp(net_physical, (xyt,), (et,))

        u, v, p = out[0], out[1], out[2]
        ux, vx, px = d_x[0], d_x[1], d_x[2]
        uy, vy, py = d_y[0], d_y[1], d_y[2]
        ut, vt = d_t[0], d_t[1]

        _, d_xx = jax.jvp(lambda z: jax.jvp(net_physical, (z,), (ex,))[1], (xyt,), (ex,))
        _, d_yy = jax.jvp(lambda z: jax.jvp(net_physical, (z,), (ey,))[1], (xyt,), (ey,))
        u_xx, v_xx = d_xx[0], d_xx[1]
        u_yy, v_yy = d_yy[0], d_yy[1]

        cont = ux + vy
        a_x = ut + u * ux + v * uy + px
        b_x = u_xx + u_yy
        a_y = vt + u * vx + v * vy + py
        b_y = v_xx + v_yy
        return cont, a_x, b_x, a_y, b_y

    return jax.vmap(single)(xyt_batch)


# =========================================================
# evaluation
# =========================================================


def eval_uvp_batch(model, x, y, t, t_max=2.0):
    def eval_single(xi, yi, ti):
        out = _net_tg_physical(model, jnp.stack([xi, yi, ti]), t_max)
        return out[0], out[1], out[2]

    return jax.vmap(eval_single)(x, y, t)


# =========================================================
# sampler
# =========================================================


def sample_interior(key, N, t_max):
    pts = jax.random.uniform(key, (N, 3))
    x = pts[:, 0] * 2.0 * jnp.pi
    y = pts[:, 1] * 2.0 * jnp.pi
    t = pts[:, 2] * t_max
    return x, y, t


# =========================================================
# loss
# =========================================================


def loss_fn_tg(model, x_col, y_col, t_col, Re, t_max=2.0):
    cont, mom_x, mom_y = residuals_batch(model, Re, x_col, y_col, t_col, t_max)
    return jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)
