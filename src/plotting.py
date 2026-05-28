import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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
    plt.close(fig)

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
    plt.close(fig)


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
    plt.close(fig)


def plot_backward_step(X, Y, u, v, p, plots_dir="plots"):
    # Figure 2 — u, v, p subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    for ax, (field, title) in zip(axes, [(u, "u (x-velocity)"), (v, "v (y-velocity)"), (p, "p (pressure)")]):
        cf = ax.contourf(X, Y, field, levels=50, cmap="RdBu_r")
        fig.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "backward_step.png"), dpi=150)
    plt.close(fig)

    # Figure 3 — velocity magnitude heatmap with normalized quiver arrows
    ny, nx = X.shape
    sx = max(1, nx // 40)
    sy = max(1, ny // 10)
    Xq = X[::sy, ::sx]
    Yq = Y[::sy, ::sx]
    Uq = u[::sy, ::sx]
    Vq = v[::sy, ::sx]
    speed_q = jnp.sqrt(Uq**2 + Vq**2)
    speed_q = jnp.where(speed_q > 0, speed_q, 1.0)
    Uq_n = Uq / speed_q
    Vq_n = Vq / speed_q

    fig2, ax2 = plt.subplots(figsize=(14, 3))
    speed = jnp.sqrt(u**2 + v**2)
    cf = ax2.contourf(X, Y, speed, levels=50, cmap="viridis")
    fig2.colorbar(cf, ax=ax2, label="|U|")
    ax2.quiver(Xq, Yq, Uq_n, Vq_n, color="white", alpha=0.85, scale=30)
    ax2.set_title("velocity magnitude + flow direction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "velocity_magnitude.png"), dpi=150)
    plt.close(fig2)


def plot_recirculation(X, Y, u, x_r, h_step, plots_dir="plots"):
    fig, ax = plt.subplots(figsize=(14, 3))

    u_np = np.array(u)

    cf = ax.contourf(X, Y, u_np, levels=50, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label="u")

    # u=0 isoline — bubble boundary
    ax.contour(X, Y, u_np, levels=[0.0], colors="black", linewidths=1.5)

    # shade recirculation region (u < 0) with hatching — avoids colour mixing
    ax.contourf(X, Y, u_np, levels=[-np.inf, 0.0],
                colors="none", hatches=["////"])

    # reattachment point
    if x_r is not None:
        ax.axvline(x_r, color="green", linestyle="--", linewidth=1.5,
                   label=f"$x_r = {x_r:.2f}$  ({x_r / h_step:.2f} H)")
        ax.plot(x_r, 0.0, "g^", markersize=8, zorder=5)

    # draw step block
    ax.add_patch(mpatches.Rectangle(
        (float(X.min()), 0.0), float(abs(X.min())), h_step,
        color="gray", zorder=4
    ))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Recirculation bubble  (hatched: u < 0,  black: u = 0 contour)")
    ax.legend()
    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(0.0, float(Y.max()))
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "recirculation.png"), dpi=150)
    plt.close(fig)


def plot_lid_cavity(X, Y, u, v, p, psi, x_vortex, y_vortex, Re, plots_dir="plots"):
    x1d = np.array(X[0, :])
    y1d = np.array(Y[:, 0])

    # ---- 2-D fields: u, v, p, streamfunction ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (field, title, cmap) in zip(
        axes.ravel(),
        [
            (u, "u  (x-velocity)", "RdBu_r"),
            (v, "v  (y-velocity)", "RdBu_r"),
            (p, "p  (pressure)", "RdBu_r"),
            (psi, r"$\psi$  (streamfunction)", "viridis"),
        ],
    ):
        cf = ax.contourf(X, Y, field, levels=60, cmap=cmap)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        if field is psi:
            ax.contour(X, Y, field, levels=20, colors="white", linewidths=0.5, alpha=0.6)
            ax.plot(x_vortex, y_vortex, "r+", markersize=12, markeredgewidth=2,
                    label=f"vortex ({x_vortex:.3f}, {y_vortex:.3f})")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle(f"Regularized lid-driven cavity  Re = {Re}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "fields.png"), dpi=150)
    plt.close(fig)

    # ---- centerline profiles ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # u along vertical centreline x = 0
    ix = np.argmin(np.abs(x1d))
    ax1.plot(np.array(u[:, ix]), y1d, color="steelblue")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("u / U_lid")
    ax1.set_ylabel("y")
    ax1.set_title("u along x = 0  (vertical centreline)")

    # v along horizontal centreline y = 0
    iy = np.argmin(np.abs(y1d))
    ax2.plot(x1d, np.array(v[iy, :]), color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v / U_lid")
    ax2.set_title("v along y = 0  (horizontal centreline)")

    fig.suptitle(f"Centreline profiles  Re = {Re}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "centerlines.png"), dpi=150)
    plt.close(fig)


def plot_Re_convergence(Re_history, Re_true, plots_dir="plots"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Re_history, color="steelblue", lw=1.5, label="Inferred Re")
    ax.axhline(Re_true, color="crimson", ls="--", lw=1.5, label=f"True Re = {Re_true:.0f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Re")
    ax.set_title("Reynolds number inference — PINN inverse problem")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "Re_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_observation_points(x_obs, y_obs, plots_dir="plots",
                            x_min=-2.0, x_max=20.0, h_step=1.0, h_chan=2.0):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(np.array(x_obs), np.array(y_obs), s=8, color="steelblue", alpha=0.7)
    step = plt.Polygon(
        [[x_min, 0], [0, 0], [0, h_step], [x_min, h_step]],
        closed=True, fc="lightgray", ec="gray",
    )
    ax.add_patch(step)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, h_chan)
    ax.set_aspect("equal")
    ax.set_title(f"Observation points (N={len(x_obs)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(plots_dir, "observation_points.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_inference_loss(loss_history, plots_dir="plots"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history, color="steelblue", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inference training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_backward_step_comparison(
    X, Y,
    u_of, v_of, p_of,
    u_pinn, v_pinn, p_pinn,
    Re,
    h_step=1.0,
    plots_dir="plots",
    filename="field_comparison.png",
    vlims=None,
):
    """3×3 comparison plot: OpenFOAM | PINN | absolute error  for u, v, p.

    Mirrors the Taylor-Green comparison layout so the two cases read identically
    in the report.

    Parameters
    ----------
    vlims : dict, optional
        Pre-computed colorbar limits for shared-range plots across Re values.
        Expected keys: 'u', 'v', 'p' → (vmin, vmax) for field columns,
                       'u_err', 'v_err', 'p_err' → (0, vmax) for error column.
        If None, each subplot uses its own local min/max.
    """
    fields = [
        (u_of, u_pinn, "u", r"$u$"),
        (v_of, v_pinn, "v", r"$v$"),
        (p_of, p_pinn, "p", r"$p$"),
    ]
    col_titles = ["OpenFOAM (reference)", "PINN (predicted)", "Absolute error"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 7))

    for row, (ref, pred, key, name) in enumerate(fields):
        ref  = np.asarray(ref)
        pred = np.asarray(pred)
        err  = np.abs(pred - ref)

        err_vmax = np.nanmax(err)

        for col, (data, cmap) in enumerate([
            (ref,  "RdBu_r"),
            (pred, "RdBu_r"),
            (err,  "hot_r"),
        ]):
            ax = axes[row, col]
            if col < 2:
                # Each subplot has its own range, but always centred at 0
                # so white = zero, blue = negative, red = positive
                vlim = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
                lvls = np.linspace(-vlim, vlim, 51)
                cf = ax.contourf(X, Y, data, levels=lvls, cmap=cmap,
                                 vmin=-vlim, vmax=vlim)
            else:
                cf = ax.contourf(X, Y, data, levels=50, cmap=cmap,
                                 vmin=0, vmax=err_vmax)
            fig.colorbar(cf, ax=ax, pad=0.02)

            ax.add_patch(mpatches.Rectangle(
                (float(X.min()), 0.0), abs(float(X.min())), h_step,
                color="0.4", zorder=5,
            ))
            ax.set_aspect("equal")
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(0.0, float(Y.max()))
            ax.set_xlabel("$x/h$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}   $y/h$", fontsize=9)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

    fig.suptitle(
        rf"Backward-facing step  —  Re = {Re:.0f}   (PINN vs. OpenFOAM)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_spatial_error_map(
    X, Y, u_pinn, u_of, Re,
    h_step=1.0, x_min=-2.0,
    plots_dir="plots",
    filename="spatial_error_map.png",
):
    """2-D heatmap of |u_pinn − u_OF| over the backward-step domain.

    Parameters
    ----------
    X, Y    : 2-D meshgrid arrays (Ny x Nx)
    u_pinn  : 2-D array — PINN u on the same grid (NaN outside fluid domain)
    u_of    : 2-D array — OpenFOAM u interpolated to the same grid
    Re      : float
    """
    error = np.abs(np.asarray(u_pinn) - np.asarray(u_of))
    X, Y  = np.asarray(X), np.asarray(Y)

    fig, ax = plt.subplots(figsize=(13, 3.5))

    cf = ax.contourf(X, Y, error, levels=50, cmap="hot_r")
    fig.colorbar(cf, ax=ax, label=r"$|u_{\mathrm{PINN}} - u_{\mathrm{OF}}|$")

    # Draw step block
    ax.add_patch(mpatches.Rectangle(
        (float(X.min()), 0.0), abs(float(X.min())), h_step,
        color="0.45", zorder=5,
    ))

    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(0.0, float(Y.max()))
    ax.set_aspect("equal")
    ax.set_xlabel("$x/h$")
    ax.set_ylabel("$y/h$")
    ax.set_title(rf"Pointwise $u$ error  |  PINN vs. OpenFOAM  —  Re = {Re:.0f}")

    fig.tight_layout()
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_lid_cavity_comparison(
    X, Y,
    u_of, v_of, p_of,
    u_pinn, v_pinn, p_pinn,
    Re,
    plots_dir="plots",
    filename="field_comparison.png",
):
    """3×3 comparison plot: OpenFOAM | PINN | absolute error  for u, v, p.
    Adapted for the square lid-driven cavity domain (no step geometry).
    """
    fields = [
        (u_of, u_pinn, r"$u$"),
        (v_of, v_pinn, r"$v$"),
        (p_of, p_pinn, r"$p$"),
    ]
    col_titles = ["OpenFOAM (reference)", "PINN (predicted)", "Absolute error"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 8))

    for row, (ref, pred, name) in enumerate(fields):
        ref  = np.asarray(ref)
        pred = np.asarray(pred)
        err  = np.abs(pred - ref)
        err_vmax = np.nanmax(err)

        for col, (data, cmap) in enumerate([
            (ref,  "RdBu_r"),
            (pred, "RdBu_r"),
            (err,  "hot_r"),
        ]):
            ax = axes[row, col]
            if col < 2:
                vlim = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
                lvls = np.linspace(-vlim, vlim, 51)
                cf = ax.contourf(X, Y, data, levels=lvls, cmap=cmap,
                                 vmin=-vlim, vmax=vlim)
            else:
                cf = ax.contourf(X, Y, data, levels=50, cmap=cmap,
                                 vmin=0, vmax=err_vmax)
            fig.colorbar(cf, ax=ax, pad=0.02)

            ax.set_aspect("equal")
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))
            ax.set_xlabel("$x$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}   $y$", fontsize=9)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

    fig.suptitle(
        rf"Lid-driven cavity  —  Re = {Re:.0f}   (PINN vs. OpenFOAM)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_comparison(
    X, Y, u_exact, v_exact, p_exact, u_pred, v_pred, p_pred, t, plots_dir="plots"
):
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
    if plots_dir is not None:
        fig.savefig(os.path.join(plots_dir, f"comparison_t{t}.png"), dpi=150)
    plt.close(fig)
