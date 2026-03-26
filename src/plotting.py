import os

import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_analytical(u, v, p, X, Y, plots_dir="plots"):
    # Plot Velocity field
    fig = plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, u, v, scale=20)
    plt.title("Taylor-Green Vortex (Velocity field)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    fig.savefig(os.path.join(plots_dir, "analytical_velocity.png"), dpi=150)
    plt.show()

    # Plot Pressure
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(
        p, extent=[0, 2 * jnp.pi, 0, 2 * jnp.pi], origin="lower", cmap="coolwarm"
    )
    plt.colorbar(label=r"Pressure ($\frac{\mathrm{N}}{\mathrm{m}^2}$)")
    plt.title("Taylor-Green Vortex (Pressure)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    fig.savefig(os.path.join(plots_dir, "analytical_pressure.png"), dpi=150)
    plt.show()


def plot_loss(adam_losses, lbfgs_losses, plots_dir="plots"):
    n_adam = len(adam_losses)
    adam_steps = list(range(n_adam))
    lbfgs_steps = list(range(n_adam, n_adam + len(lbfgs_losses)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(adam_steps, adam_losses, label="Adam", color="steelblue")
    ax.semilogy(lbfgs_steps, lbfgs_losses, label="L-BFGS", color="darkorange")
    ax.axvline(n_adam, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.show()


def plot_comparison(X, Y, u_exact, v_exact, p_exact, u_pred, v_pred, p_pred, t, plots_dir="plots"):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fields = [(u_exact, u_pred, "u"), (v_exact, v_pred, "v"), (p_exact, p_pred, "p")]
    for row, (exact, pred, name) in enumerate(fields):
        err = jnp.abs(exact - pred)
        axes[row, 0].contourf(X, Y, exact, levels=50, cmap="RdBu_r")
        axes[row, 0].set_title(f"{name} exact, t={t}")
        axes[row, 1].contourf(X, Y, pred, levels=50, cmap="RdBu_r")
        axes[row, 1].set_title(f"{name} predicted, t={t}")
        cf = axes[row, 2].contourf(X, Y, err, levels=50, cmap="hot_r")
        axes[row, 2].set_title(f"{name} absolute error, t={t}")
        fig.colorbar(cf, ax=axes[row, 2])

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"comparison_t{t}.png"), dpi=150)
    plt.show()
