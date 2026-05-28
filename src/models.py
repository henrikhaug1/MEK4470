import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax


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
# Generic PINN model (used by backward step and lid cavity)
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
# Taylor-Green PINN model (Fourier features, 5 inputs)
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
# SIREN model (sine activation with proper initialization)
#
# Key differences from BackwardStepPINN:
#   - First layer weights scaled by omega_0 (default 30) so the network
#     sees high-frequency input signals from the start
#   - Hidden layer weights drawn from U(-sqrt(6/fan_in), sqrt(6/fan_in))
#     (Sitzmann et al. 2020 initialization)
#   - Activation is always jnp.sin — passed in for interface compatibility
#     but overridden internally
# =========================================================


# =========================================================
# Fourier-feature PINN (drop-in replacement for BackwardStepPINN)
#
# The only change vs. BackwardStepPINN:
#   Input (x_n, y_n) is mapped through a fixed random Fourier projection
#   before entering the tanh MLP:
#       φ = [sin(2π B xy), cos(2π B xy)]   B ∈ ℝ^{n_fourier × 2}  (fixed)
#
#   This directly counteracts spectral bias — the MLP now sees a rich
#   frequency representation of the input rather than two scalars.
#   (Tancik et al., 2020; "Fourier Features Let Networks Learn
#    High Frequency Functions in Low Dimensional Domains")
#
#   B is a plain JAX array (not nnx.Param) so it is NEVER updated by
#   the optimizer.  It lives in the graphdef and is preserved through
#   nnx.split / nnx.merge correctly.
# =========================================================


class FourierBackwardStepPINN(nnx.Module):
    def __init__(
        self,
        widths,
        *,
        key,
        activation=jax.nn.tanh,
        n_inputs: int = 2,
        n_fourier: int = 16,
        sigma: float = 1.0,
    ):
        self.activation = activation
        self.n_fourier = n_fourier
        self.n_inputs_orig = n_inputs
        self.sigma = sigma

        k1, k2 = jax.random.split(key)
        # Fixed projection matrix — plain array, not nnx.Param
        self._B = jax.random.normal(k1, (n_inputs, n_fourier)) * sigma  # (2, n_fourier)

        # Standard MLP but with 2·n_fourier inputs
        dims = [2 * n_fourier] + list(widths) + [3]
        keys = jax.random.split(k2, len(dims))
        self.ws = nnx.List()
        self.bs = nnx.List()
        for k, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, _ = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / (din + dout))
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(jnp.zeros((dout,))))

    def __call__(self, xy):
        # xy: (..., 2) — works for (1, 2) and (2,) inputs
        proj = xy @ self._B                                            # (..., n_fourier)
        phi  = jnp.concatenate(
            [jnp.sin(2.0 * jnp.pi * proj),
             jnp.cos(2.0 * jnp.pi * proj)], axis=-1)                  # (..., 2·n_fourier)
        z = phi
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = self.activation(z @ W + b)
        return z @ self.ws[-1] + self.bs[-1]


def save_fourier_model(model, path):
    """Save FourierBackwardStepPINN — includes fixed projection matrix B."""
    state = nnx.state(model, nnx.Param)
    flat, _ = jax.tree.flatten(state)
    arrays = {f"p{i}": np.array(a) for i, a in enumerate(flat)}
    n_layers = len(model.ws)
    widths = np.array([int(model.ws[i].value.shape[1]) for i in range(n_layers - 1)])
    np.savez(
        path,
        _n=np.array(len(flat)),
        _widths=widths,
        _n_fourier=np.array(model.n_fourier),
        _sigma=np.array(model.sigma),
        _B=np.array(model._B),
        **arrays,
    )


def load_fourier_model(path, *, key, activation=jax.nn.tanh):
    """Reconstruct a FourierBackwardStepPINN from a saved .npz file."""
    data = np.load(path)
    widths = list(data["_widths"])
    model = FourierBackwardStepPINN(
        widths,
        key=key,
        activation=activation,
        n_fourier=int(data["_n_fourier"]),
        sigma=float(data["_sigma"]),
    )
    # Restore the exact projection matrix used during training
    model._B = jnp.array(data["_B"])
    load_model_state(model, path)
    return model


class SirenPINN(nnx.Module):
    def __init__(self, widths, *, key, activation=jnp.sin, n_inputs=2, omega_0=30.0):
        dims = [n_inputs] + list(widths) + [3]
        keys = jax.random.split(key, len(dims))
        self.ws = nnx.List()
        self.bs = nnx.List()
        self.omega_0 = omega_0

        for layer, (k, din, dout) in enumerate(zip(keys, dims[:-1], dims[1:])):
            wk, bk = jax.random.split(k)
            if layer == 0:
                # First layer: uniform in [-1/fan_in, 1/fan_in], scaled by omega_0
                W = jax.random.uniform(wk, (din, dout), minval=-1.0 / din, maxval=1.0 / din) * omega_0
            elif layer < len(dims) - 2:
                # Hidden layers: U(-sqrt(6/fan_in), sqrt(6/fan_in))
                limit = jnp.sqrt(6.0 / din)
                W = jax.random.uniform(wk, (din, dout), minval=-limit, maxval=limit)
            else:
                # Output layer: Xavier (same as BackwardStepPINN)
                W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / (din + dout))
            b = jnp.zeros((dout,))
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, xy):
        z = xy
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jnp.sin(z @ W + b)
        return z @ self.ws[-1] + self.bs[-1]
