import jax
import jax.numpy as jnp


# =========================================================
# hard BC ansatz
# =========================================================


def hard_bc_ansatz_lc(x, y, u_raw, v_raw, p_raw):
    lid = (1.0 - x**2) ** 2
    u = lid * (1.0 + y) / 2.0 + (1.0 - y**2) * (1.0 - x**2) * u_raw
    v = (1.0 - y**2) * (1.0 - x**2) * v_raw
    return u, v, p_raw


# =========================================================
# physics residuals
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


def residual_parts_lc(model, x, y):
    """Split the lid-cavity momentum residual into Re-independent and viscous parts.

    Returns (cont, a_x, b_x, a_y, b_y) where, per momentum component,
        mom = a - nu * b,   nu = 2/Re,
    with
        a = u·∇u + ∇p   (inertial + pressure, independent of Re)
        b = ∇²u         (Laplacian; viscous term is nu·b)

    The optimal viscosity is then the least-squares projection
        nu* = Σ(a·b)/Σ(b·b)   ->   Re = 2/nu*.
    """
    ex = jnp.array([1.0, 0.0])
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack([x, y], axis=-1)

    def single(xy):
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

        cont = ux + vy
        a_x = u * ux + v * uy + px
        b_x = u_xx + u_yy
        a_y = u * vx + v * vy + py
        b_y = v_xx + v_yy
        return cont, a_x, b_x, a_y, b_y

    return jax.vmap(single)(xy_batch)


# =========================================================
# evaluation
# =========================================================


def eval_uvp_batch_lc(model, x, y):
    def eval_single(xi, yi):
        raw = model(jnp.stack([xi, yi])[None])[0]
        return hard_bc_ansatz_lc(xi, yi, raw[0], raw[1], raw[2])

    return jax.vmap(eval_single)(x, y)


# =========================================================
# sampler
# =========================================================


def sample_interior_lc(key, N):
    pts = jax.random.uniform(key, (N, 2)) * 2.0 - 1.0
    return pts[:, 0], pts[:, 1]
