from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# Small positive floor used as the lower bound for pressure/density in the
# bounded nonlinear solve. This replaces the previous approach of solving
# unconstrained and then patching negative results with np.abs() after the
# fact -- here the solver is simply never allowed to visit a non-physical
# state in the first place, so sqrt(p/rho) is always well defined during
# the solve itself, not just after clean-up.
POSITIVITY_FLOOR = 1e-10


def dump_results_binary(filename, t, r, rho, rhou, rhoE):
    with open(filename, "ab") as f:
        # Save time as a 1D array, then stack the state variables into a 2D array
        np.save(f, np.array([t], dtype=np.float64))
        np.save(f, np.vstack((r, rho, rhou, rhoE)))


def read_results_binary(filename):
    results = []

    with open(filename, "rb") as f:
        while True:
            try:
                # Load time array
                t = np.load(f)[0]
                # Load stacked variables
                data = np.load(f)

                r = data[0]
                rho = data[1]
                rhou = data[2]
                rhoE = data[3]

                results.append((t, r, rho, rhou, rhoE))
            except (EOFError, ValueError):
                # Reached end of file
                break
    return results


def simulate_zouh_moc(filename="results.bin", dt_save=0.02, cfl=0.5, print_every=50):
    # --- Zouh Initial Conditions ---
    gamma = 1.4
    l_0 = 1.0  # Domain radius
    blast_radius = 0.13
    t_fin = 0.13
    j_geom = 1  # 1 = cylindrical, 2 = spherical

    rho_L, vel_L, pres_L = 2.0, 0.0, 15.0
    rho_R, vel_R, pres_R = 1.0, 0.0, 2.0

    # Grid setup (Note: N=1000 will take a little while to compute with pure MOC!)
    N = 1000
    r_grid = np.linspace(0, l_0, N)
    dr = l_0 / (N - 1)

    # Initialize primitive fields
    rho = np.where(r_grid < blast_radius, rho_L, rho_R)
    p = np.where(r_grid < blast_radius, pres_L, pres_R)
    u = np.where(r_grid < blast_radius, vel_L, vel_R)

    def max_wave_speed(u_field, p_field, rho_field):
        # Recomputed from the *current* state every step, rather than frozen
        # once from the initial left state. The initial left-state estimate
        # only bounds the wave speed at t=0; downstream of the shock/rarefaction
        # interaction the true local wave speed can exceed it, silently
        # breaking the CFL condition for the remainder of the run if dt is
        # never revisited.
        a_field = np.sqrt(gamma * p_field / rho_field)
        return np.max(np.abs(u_field) + a_field)

    t = 0.0
    dt = cfl * dr / max_wave_speed(u, p, rho)

    # --- Prepare File & Save Initial State ---
    if os.path.exists(filename):
        os.remove(filename)

    next_save_t = 0.0
    rhou_init = rho * u
    rhoE_init = p / (gamma - 1.0) + 0.5 * rho * u**2
    dump_results_binary(filename, t, r_grid, rho, rhou_init, rhoE_init)
    next_save_t += dt_save

    # Helper: Interpolate old state with reflective boundary at r=0
    def get_old_state(r_query, u_old, p_old, rho_old):
        if r_query < 0:
            # Ghost node logic: reflect position and invert velocity
            r_eval = -r_query
            u_val = -np.interp(r_eval, r_grid, u_old)
        else:
            r_eval = r_query
            u_val = np.interp(r_eval, r_grid, u_old)

        # P and Rho are scalar/symmetric across origin
        p_val = np.interp(r_eval, r_grid, p_old)
        rho_val = np.interp(r_eval, r_grid, rho_old)
        a_val = np.sqrt(gamma * p_val / rho_val)
        return u_val, p_val, rho_val, a_val

    print(f"Starting pure MOC simulation. Target time: {t_fin}")

    step_count = 0
    while t < t_fin:
        if t + dt > t_fin:
            dt = t_fin - t

        u_new = np.zeros(N)
        p_new = np.zeros(N)
        rho_new = np.zeros(N)

        # --- Solve each node using MOC Compatibility Equations ---
        for i in range(N):
            r_i = r_grid[i]

            if i == 0:

                def residual_origin(vars):
                    p_guess, rho_guess = vars
                    a_guess = np.sqrt(gamma * p_guess / rho_guess)

                    r_minus = r_i - (-a_guess) * dt
                    r_zero = r_i

                    u_m, p_m, rho_m, a_m = get_old_state(r_minus, u, p, rho)
                    _, p_0, rho_0, _ = get_old_state(r_zero, u, p, rho)

                    rhoa_m = 0.5 * (rho_guess * a_guess + rho_m * a_m)

                    S_m = j_geom * a_m * u_m / r_minus if r_minus > 1e-6 else 0.0

                    res1 = (0.0 - u_m) - (p_guess - p_m) / rhoa_m - dt * S_m
                    res2 = p_guess / (rho_guess**gamma) - p_0 / (rho_0**gamma)
                    return [res1, res2]

                sol = least_squares(
                    residual_origin,
                    [p[0], rho[0]],
                    bounds=([POSITIVITY_FLOOR, POSITIVITY_FLOOR], [np.inf, np.inf]),
                    method="trf",
                )
                u_new[0] = 0.0
                p_new[0], rho_new[0] = sol.x

            else:

                def residual_interior(vars):
                    u_guess, p_guess, rho_guess = vars
                    a_guess = np.sqrt(gamma * p_guess / rho_guess)

                    r_plus = r_i - (u_guess + a_guess) * dt
                    r_minus = r_i - (u_guess - a_guess) * dt
                    r_zero = r_i - u_guess * dt

                    u_p, p_p, rho_p, a_p = get_old_state(r_plus, u, p, rho)
                    u_m, p_m, rho_m, a_m = get_old_state(r_minus, u, p, rho)
                    _, p_0, rho_0, _ = get_old_state(r_zero, u, p, rho)

                    rhoa_p = 0.5 * (rho_guess * a_guess + rho_p * a_p)
                    rhoa_m = 0.5 * (rho_guess * a_guess + rho_m * a_m)

                    S_p = -j_geom * a_p * u_p / r_plus if r_plus > 1e-6 else 0.0
                    S_m = j_geom * a_m * u_m / r_minus if r_minus > 1e-6 else 0.0

                    res1 = (u_guess - u_p) + (p_guess - p_p) / rhoa_p - dt * S_p
                    res2 = (u_guess - u_m) - (p_guess - p_m) / rhoa_m - dt * S_m
                    res3 = p_guess / (rho_guess**gamma) - p_0 / (rho_0**gamma)
                    return [res1, res2, res3]

                sol = least_squares(
                    residual_interior,
                    [u[i], p[i], rho[i]],
                    bounds=(
                        [-np.inf, POSITIVITY_FLOOR, POSITIVITY_FLOOR],
                        [np.inf, np.inf, np.inf],
                    ),
                    method="trf",
                )
                u_new[i], p_new[i], rho_new[i] = sol.x

        # Update state (no np.abs() patching needed: the bounded solve above
        # never returns negative p/rho in the first place)
        u, p, rho = u_new, p_new, rho_new
        t += dt
        step_count += 1

        # Recompute dt from the *updated* state for the next step, so the
        # CFL condition is enforced continuously rather than only at t=0.
        dt = cfl * dr / max_wave_speed(u, p, rho)

        if step_count % print_every == 0 or np.isclose(t, t_fin):
            print(f"Time : {t:.5f}  (dt = {dt:.3e})")

        # --- Check for save ---
        if t >= next_save_t or np.isclose(t, t_fin):
            rhou_save = rho * u
            rhoE_save = p / (gamma - 1.0) + 0.5 * rho * u**2
            dump_results_binary(filename, t, r_grid, rho, rhou_save, rhoE_save)
            next_save_t += dt_save

    print("Simulation complete.")


if __name__ == "__main__":
    file_path = Path("build/moc_radial")

    file_path.mkdir(parents=True, exist_ok=True)

    file_path = Path(file_path / "zouh_moc_results.bin")

    # 1. Run simulation and dump to file
    simulate_zouh_moc(filename=file_path, dt_save=0.02)

    # 2. Read results back from the file
    print(f"Reading results from {file_path}...")
    saved_data = read_results_binary(file_path)

    # 3. Extract the final time step
    final_t, r, rho, rhou, rhoE = saved_data[-1]
    print(f"Plotting state at t = {final_t:.5f}")

    # Recover primitive variables for plotting
    gamma = 1.4
    u = rhou / rho
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * u**2)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        f"MOC Solver: Zouh Radial Riemann Problem at $t={final_t:.3f}$", fontsize=14
    )

    axs[0].plot(r, rho, "b-", label="Density ($\\rho$)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(r, u, "r-", label="Velocity ($u$)")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(r, p, "g-", label="Pressure ($p$)")
    axs[2].legend()
    axs[2].grid()
    axs[2].set_xlabel("Radius (r)")

    plt.tight_layout()
    plt.savefig("build/moc_radial/zouh_moc_plot.png", dpi=150)
    print("Saved plot to build/moc_radial/zouh_moc_plot.png")
