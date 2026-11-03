from pathlib import Path
import matplotlib.pyplot as plt


def plot_flow_fields(X_glob, Y_glob, fields, l_0, save_path=None, show=True, dpi=300):
    """Plot 2D hydrodynamic flow field variables on a 2x2 grid.

    Parameters
    ----------
    X_glob, Y_glob : np.ndarray
        Spatial grid coordinates.
    fields : dict
        Mapping of subplot keys/titles to 2D field arrays.
    l_0 : float
        Domain upper limit for both axes.
    save_path : str, optional
        File path to save the generated figure.
    show : bool, default=True
        Whether to display the plot interactively.
    dpi : int, default=300
        Resolution for exported figure.
    """

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), constrained_layout=True, sharex=True, sharey=True
    )

    plot_configs = [
        (axes[0, 0], fields["rho"], r"Density ($\rho$)", r"$\rho$"),
        (
            axes[0, 1],
            fields["rhou"],
            r"Momentum $x$ ($\rho u$)",
            r"$\rho u$",
        ),
        (
            axes[1, 0],
            fields["rhov"],
            r"Momentum $y$ ($\rho v$)",
            r"$\rho v$",
        ),
        (
            axes[1, 1],
            fields["rhoE"],
            r"Total Energy ($\rho E$)",
            r"$\rho E$",
        ),
    ]

    for ax, data, title, cbar_label in plot_configs:
        cf = ax.contourf(X_glob, Y_glob, data, levels=100, cmap="turbo")

        cbar = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.04)
        cbar.set_label(cbar_label)

        ax.set_xlim(0, l_0)
        ax.set_ylim(0, l_0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="medium")

    # Outer-axis labels for shared grid layout
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$x$")

    fig.suptitle(r"2D shock test. Flow field contours", fontsize=14, fontweight="bold")

    if save_path:
        print(f"Saving image at:{save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return
