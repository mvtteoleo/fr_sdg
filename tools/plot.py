import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi, legendre
import sys
import os

# --- Your Simulation Config ---


def parse_petsc_opts(filepath):
    """Reads a PETSc .opts file and extracts mesh parameters."""
    # Default fallback values
    params = {
        "n_elems": 20,
        "poly_deg": 4,
        "domain_length": 10.0,
        "sol_file": "conf_files/acoustic_rk.opts",
    }

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using defaults: {params}")
        return params

    with open(filepath, "r") as f:
        for line in f:
            # Strip whitespace and ignore comments or empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split the line by spaces (e.g., "-n_elems 60" -> ["-n_elems", "60"])
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "-n_elems":
                    params["n_elems"] = int(parts[1])
                elif parts[0] == "-poly_deg":
                    params["poly_deg"] = int(parts[1])
                elif parts[0] == "-box_l":
                    params["domain_length"] = float(parts[1])
                elif parts[0] == "-ts_monitor_solution":
                    raw_path = parts[1]
                    if raw_path.startswith("binary:"):
                        raw_path = raw_path.split("binary:", 1)[1]

                    params["sol_file"] = raw_path.strip()

    return params


def map(xi, a, b):
    return xi * (b - a) / 2 + (a + b) / 2


def csi_ws(poly_deg):
    """
    Computes Gauss-Lobatto-Legendre (GLL) nodes (csi) and weights (w_s)
    on the reference element [-1, 1].

    n_points: Total number of nodes (polynomial degree + 1)
    """
    n_points = poly_deg + 1

    # 1. Initialize arrays
    csi = np.zeros(n_points)
    w_s = np.zeros(n_points)

    if n_points == 1:
        csi = np.array([0.0])
        w_s = np.array([2.0])

    if n_points == 2:
        csi = np.array([-1.0, +1.0])
        w_s = np.array([1.0, 1.0])

    elif n_points > 2:
        # 2. Set the boundary nodes
        csi[0] = -1.0
        csi[-1] = 1.0

        # 3. Compute interior nodes
        # Interior nodes of GLL are the roots of the Jacobi polynomial P^(1,1)_{N-2}(x)
        if n_points > 2:
            csi[1:-1], _ = roots_jacobi(n_points - 2, 1, 1)

        # 4. Compute the weights
        # Formula: w_i = 2 / ( N * (N-1) * [P_{N-1}(x_i)]^2 )
        # where N is n_points, and P_{N-1} is the Legendre polynomial of degree N-1.
        poly_degree = n_points - 1
        P_N = legendre(poly_degree)

        for i in range(n_points):
            w_s[i] = 2.0 / (n_points * poly_degree * (P_N(csi[i]) ** 2))
    return csi, w_s


def get_mesh_on_gll(n_elems, poly_deg, domain_length):
    n_nds = poly_deg + 1

    x_pts = np.ndarray([n_elems * n_nds])
    csi, _ = csi_ws(poly_deg)
    h = domain_length / n_elems
    for el in range(n_elems):
        elem_x0 = float(el * h)
        for node in range(n_nds):
            x_pts[el * n_nds + node] = map(csi[node], elem_x0, elem_x0 + h)
    return x_pts


def read_all_petsc_vectors(filename):
    """Reads a PETSc binary file containing multiple appended time steps."""
    vectors = []
    try:
        with open(filename, "rb") as f:
            while True:
                header = np.fromfile(f, dtype=">i4", count=2)
                if header.size == 0:
                    break

                classid, size = header[0], header[1]
                if classid != 1211214:
                    f.seek(-8, 1)
                    header = np.fromfile(f, dtype=">i8", count=2)
                    classid, size = header[0], header[1]

                data = np.fromfile(f, dtype=">f8", count=size)
                vectors.append(data)
    except FileNotFoundError:
        pass
    return vectors


if __name__ == "__main__":
    # Grab the file

    # --- Setup Configuration from Command Line ---
    # Expected usage: python3 plot.py acoustic_rk.opts
    opts_file = sys.argv[1] if len(sys.argv) > 1 else "acoustic_rk.opts"

    # Extract the parameters automatically!
    config = parse_petsc_opts(opts_file)
    target_file = config["sol_file"]
    n_elems = config["n_elems"]
    poly_deg = config["poly_deg"]
    domain_length = 10.0
    nds = poly_deg + 1

    print(f"Loaded config from {opts_file}: n_elems={n_elems}, poly_deg={poly_deg}")

    if not os.path.exists(target_file):
        print(f"File {target_file} not found!")
        sys.exit(1)

    print(f"Parsing {target_file}...")
    all_steps = read_all_petsc_vectors(target_file)
    num_steps = len(all_steps)
    num_points = n_elems * nds
    print(f"Successfully extracted {num_steps} time steps!")

    # --- Build the History Matrices for ALL Variables ---
    rho_history = np.zeros((num_steps, num_points))
    rhoU_history = np.zeros((num_steps, num_points))
    rhoE_history = np.zeros((num_steps, num_points))

    for i, step_data in enumerate(all_steps):
        data_3d = step_data.reshape((n_elems, 3, nds))

        # Extract each equation and flatten across the domain
        rho_history[i, :] = data_3d[:, 0, :].flatten()
        rhoU_history[i, :] = data_3d[:, 1, :].flatten()
        rhoE_history[i, :] = data_3d[:, 2, :].flatten()

    x_coords = get_mesh_on_gll(n_elems, poly_deg, domain_length)

    # ==========================================================
    # PLOT: 3-Panel Snapshot Overlay
    # ==========================================================
    # Create a figure with 3 subplots stacked vertically, sharing the X-axis
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    just_4 = [0, num_steps // 3, 2 * num_steps // 3, num_steps - 1]

    for step_idx in np.linspace(0, num_steps - 1, 10):  # range(0, num_steps):
        # for step_idx in range(0, num_steps):
        step_idx = int(step_idx)
        label = f"Step {step_idx}" if step_idx != 0 else "Initial (t=0)"

        # Plot Density
        axs[0].plot(x_coords, rho_history[step_idx, :], alpha=0.8, label=label)
        # Plot Momentum
        axs[1].plot(x_coords, rhoU_history[step_idx, :], alpha=0.8)
        # Plot Energy
        axs[2].plot(x_coords, rhoE_history[step_idx, :], alpha=0.8)

    # --- Formatting ---
    fig.suptitle("1D Euler: Isentropic Acoustic Pulse", fontsize=16)

    # Density Axis
    axs[0].set_ylabel(r"Density ($\rho$)")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(loc="upper right")

    # Momentum Axis
    axs[1].set_ylabel(r"Momentum ($\rho U$)")
    axs[1].grid(True, linestyle="--", alpha=0.7)

    # Energy Axis
    axs[2].set_ylabel(r"Total Energy ($\rho E$)")
    axs[2].set_xlabel("Position (x)")
    axs[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
