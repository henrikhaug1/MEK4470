import jax.numpy as jnp


# Analytical solutions for benchmarking:
def taylor_green_u(x, y, t, nu):
    return -jnp.cos(x) * jnp.sin(y) * jnp.exp(-2 * nu * t)


def taylor_green_v(x, y, t, nu):
    return jnp.sin(x) * jnp.cos(y) * jnp.exp(-2 * nu * t)


def taylor_green_p(x, y, t, nu):
    return -0.25 * (jnp.cos(2 * x) + jnp.cos(2 * y)) * jnp.exp(-4 * nu * t)
