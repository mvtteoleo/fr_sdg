import subprocess
import math
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
EXEC_CMD = ["mpirun", "-n", "3", "./build/test/acoustic_wave"]
BASE_ARGS = ["-options_file", "conf_files/acoustic_rk.opts"]


def print_convs(order_rho, order_rhou, order_rhoe) -> None:
    print(f"Convergence orders:")
    print(rf"Rho {float(order_rho)}")
    print(rf"RhoU {float(order_rhou)}")
    print(rf"RhoE {float(order_rhoe)}")
    return


def run_sim_get_errs(n_elems, poly_deg, dt):
    """Runs the C++ solver overriding specific PETSc flags and extracts the L2 errors."""

    cmd = (
        EXEC_CMD
        + BASE_ARGS
        + ["-n_elems", str(n_elems), "-poly_deg", str(poly_deg), "-ts_dt", str(dt)]
    )

    print(f"Running: N={n_elems:<3} | P={poly_deg} | dt={dt:<7} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    m_rho = re.search(r"Rho\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
    m_rhou = re.search(r"RhoU\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
    m_rhoe = re.search(r"RhoE\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)

    print(f"Errori: ")
    print(f"\t Rho : {float(m_rho.group(1) )}")
    print(f"\t RhoU: {float(m_rhou.group(1))}")
    print(f"\t RhoE: {float(m_rhoe.group(1))}")
    if m_rho and m_rhou and m_rhoe:
        rho_val = float(m_rho.group(1))

        # Catch the explosion gracefully!
        if math.isnan(rho_val):
            print(
                f"  -> [EXPLOSION] Simulation went unstable (CFL violation) at dt={dt}"
            )

        return rho_val, float(m_rhou.group(1)), float(m_rhoe.group(1))
    else:
        print("ERROR: Could not find L2 errors in output! Dumping stdout:")
        print(result.stdout)
        return None, None, None


def get_convergence_order(x_array, y_array, is_spatial):
    """Calculates the numerical order of convergence using linear regression on log-log data."""
    # Fit a line: log(Error) = slope * log(x) + intercept
    slope, _ = np.polyfit(np.log(x_array), np.log(y_array), 1)

    # For spatial (x = N_elems), error ~ (1/N)^p -> slope is negative. We want the absolute value.
    # For temporal (x = dt), error ~ dt^p -> slope is positive.
    return abs(slope) if is_spatial else slope


# ==========================================
# TEST 1: Spatial h-refinement (Vary N_elems)
# ==========================================
def test_h_refinement():
    print("\n" + "=" * 40)
    print(" SPATIAL h-REFINEMENT TEST")
    print("=" * 40)

    # P=2 means theoretical order is O(h^3).
    # dt=0.0001 ensures temporal error stays below 1e-8, avoiding the floor.
    fixed_dt = 0.0001
    fixed_p = 2
    N_array = np.array([3 * (n**2) for n in range(1, 4)])

    errs_rho, errs_rhou, errs_rhoe = [], [], []

    for N in N_array:
        rho, rhou, rhoe = run_sim_get_errs(N, fixed_p, fixed_dt)
        errs_rho.append(rho)
        errs_rhou.append(rhou)
        errs_rhoe.append(rhoe)

    # Calculate Numerical Orders
    order_rho = get_convergence_order(N_array, errs_rho, is_spatial=True)
    order_rhou = get_convergence_order(N_array, errs_rhou, is_spatial=True)
    order_rhoe = get_convergence_order(N_array, errs_rhoe, is_spatial=True)
    print_convs(order_rho, order_rhou, order_rhoe)

    # Plotting
    plt.figure(figsize=(9, 6))
    plt.loglog(N_array, errs_rho, marker="o", label=f"Rho  (Order: {order_rho:.2f})")
    plt.loglog(N_array, errs_rhou, marker="s", label=f"RhoU (Order: {order_rhou:.2f})")
    plt.loglog(N_array, errs_rhoe, marker="^", label=f"RhoE (Order: {order_rhoe:.2f})")

    # Reference Line O(h^(P+1))
    ref_line = errs_rho[0] * (N_array[0] / N_array) ** (fixed_p + 1)
    plt.loglog(
        N_array,
        ref_line,
        linestyle="--",
        color="black",
        label=f"Theoretical O(h^{fixed_p+1})",
    )

    plt.title(f"Spatial Convergence (Fixed P={fixed_p}, dt={fixed_dt})")
    plt.xlabel("Number of Elements (N)")
    plt.ylabel("Continuous L2 Error")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


# ==========================================
# TEST 2: Temporal Convergence (Vary dt)
# ==========================================
def test_dt_refinement():
    print("\n" + "=" * 40)
    print(" TEMPORAL dt-REFINEMENT TEST")
    print("=" * 40)

    # Massive spatial mesh (N=300, P=4) pushes spatial error below ~1e-9.
    # This gives RK2 a massive runway to show perfect O(dt^2) convergence.
    fixed_N = 500
    fixed_p = 2
    dt_array = np.array([0.1 * 0.5**n for n in range(4)])

    errs_rho, errs_rhou, errs_rhoe = [], [], []

    for dt in dt_array:
        rho, rhou, rhoe = run_sim_get_errs(fixed_N, fixed_p, dt)
        errs_rho.append(rho)
        errs_rhou.append(rhou)
        errs_rhoe.append(rhoe)

    # Calculate Numerical Orders
    order_rho = get_convergence_order(dt_array, errs_rho, is_spatial=False)
    order_rhou = get_convergence_order(dt_array, errs_rhou, is_spatial=False)
    order_rhoe = get_convergence_order(dt_array, errs_rhoe, is_spatial=False)
    print_convs(order_rho, order_rhou, order_rhoe)

    # Plotting
    plt.figure(figsize=(9, 6))
    plt.loglog(dt_array, errs_rho, marker="o", label=f"Rho  (Order: {order_rho:.2f})")
    plt.loglog(dt_array, errs_rhou, marker="s", label=f"RhoU (Order: {order_rhou:.2f})")
    plt.loglog(dt_array, errs_rhoe, marker="^", label=f"RhoE (Order: {order_rhoe:.2f})")

    # Reference Line O(dt^2)
    ref_line = errs_rho[0] * (dt_array / dt_array[0]) ** 2
    plt.loglog(
        dt_array, ref_line, linestyle="--", color="black", label="Theoretical O(dt^2)"
    )

    plt.title(f"Temporal Convergence (Fixed N={fixed_N}, P={fixed_p})")
    plt.xlabel("Time Step (dt)")
    plt.ylabel("Continuous L2 Error")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


# ==========================================
# TEST 3: p-Convergence (Vary P)
# ==========================================
def test_p_refinement():
    print("\n" + "=" * 40)
    print(" p-REFINEMENT TEST")
    print("=" * 40)

    # Massive spatial mesh (N=300, P=4) pushes spatial error below ~1e-9.
    # This gives RK2 a massive runway to show perfect O(dt^2) convergence.
    fixed_N = 500
    fixed_dt = 0.0001
    p_array = np.array([1, 2, 3, 4])

    errs_rho, errs_rhou, errs_rhoe = [], [], []

    for p in p_array:
        rho, rhou, rhoe = run_sim_get_errs(fixed_N, p, fixed_dt)
        errs_rho.append(rho)
        errs_rhou.append(rhou)
        errs_rhoe.append(rhoe)

    # Calculate Numerical Orders
    order_rho = get_convergence_order(p_array, errs_rho, is_spatial=False)
    order_rhou = get_convergence_order(p_array, errs_rhou, is_spatial=False)
    order_rhoe = get_convergence_order(p_array, errs_rhoe, is_spatial=False)
    print_convs(order_rho, order_rhou, order_rhoe)

    # Plotting
    plt.figure(figsize=(9, 6))
    plt.loglog(p_array, errs_rho, marker="o", label=f"Rho  (Order: {order_rho:.2f})")
    plt.loglog(p_array, errs_rhou, marker="s", label=f"RhoU (Order: {order_rhou:.2f})")
    plt.loglog(p_array, errs_rhoe, marker="^", label=f"RhoE (Order: {order_rhoe:.2f})")

    # Reference Line O(dt^2)
    ref_line = errs_rho[0] * (p_array / p_array[0]) ** -2
    plt.loglog(
        p_array, ref_line, linestyle="--", color="black", label="Theoretical O(dt^2)"
    )

    plt.title(f"Temporal Convergence (Fixed N={fixed_N}, dt={fixed_dt})")
    plt.xlabel("Poly-degree (P)")
    plt.ylabel("Continuous L2 Error")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.gca().invert_xaxis()  # Smaller dt on the right is standard practice
    plt.show()


if __name__ == "__main__":
    # Ensure your C++ code is compiled first!
    print("Compiling latest C++ source...")
    # subprocess.run(["make", "euler_1d"], check=True)

    test_h_refinement()
    test_dt_refinement()
    test_p_refinement()
