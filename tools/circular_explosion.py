import re
import json
from pathlib import Path


def summarize_tree_results(base_dir: str | Path, target_t_f: float = 0.13):
    """
    Iterates over the directory tree and prints a CLI table with N, P, Scheme, dt,
    Worked status, and Time. Prioritizes meta.json over run.log.
    """
    base_dir = Path(base_dir)
    print("\n" + "=" * 80)
    print(" DIRECTORY TREE SIMULATION SUMMARY")
    print("=" * 80)

    results = []
    evaluated_dirs = set()

    # Iterate over all directories (not just files) to catch everything
    for current_dir in base_dir.rglob("*"):
        if not current_dir.is_dir():
            continue

        meta_json = current_dir / "meta.json"
        run_log = current_dir / "run.log"

        # If neither file exists here, it's not a run directory
        if not (meta_json.exists() or run_log.exists()):
            continue

        # Avoid double-counting
        if current_dir in evaluated_dirs:
            continue
        evaluated_dirs.add(current_dir)

        path_str = str(current_dir)

        # 1. Parse N and P from the directory path
        np_match = re.search(r"n_(\d+)_p_(\d+)", path_str)
        if not np_match:
            continue
        n_elems = int(np_match.group(1))
        p_deg = int(np_match.group(2))

        # 2. Parse Scheme (Implicit vs Explicit)
        is_impl = "BFD2" in path_str or "BDF2" in path_str or "bdf2" in path_str
        scheme = "Implicit" if is_impl else "Explicit"

        # 3. Parse dt from the folder name
        try:
            dt = float(current_dir.name)
        except ValueError:
            dt = -1.0

        worked = False
        time_sol = 0.0

        # 4. PRIMARY DATA SOURCE: Check meta.json first
        if meta_json.exists():
            try:
                with open(meta_json, "r") as f:
                    meta = json.load(f)

                time_sol = meta.get("time_to_sol", 0.0)
                if dt < 0:
                    dt = meta.get("dt", -1.0)

                # Handle your tuple logic: check_sim_ended returns (bool, [times])
                did_end = meta.get("did_end", False)
                if isinstance(did_end, (list, tuple)):
                    worked = bool(did_end[0])
                else:
                    worked = bool(did_end)
            except Exception:
                pass

        # 5. BACKUP DATA SOURCE: If meta.json was missing or failed, parse run.log
        if run_log.exists() and (not worked or time_sol == 0.0):
            try:
                log_content = run_log.read_text(encoding="utf-8", errors="ignore")

                if not worked:
                    times = []
                    for line in log_content.splitlines():
                        match = re.search(r"Time\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", line)
                        if match:
                            times.append(float(match.group(1)))
                    if times:
                        last_time = times[-1]
                        if (
                            target_t_f >= last_time
                            and abs(target_t_f - last_time) <= 1e-8
                        ):
                            worked = True
                        elif last_time >= target_t_f:
                            worked = True

                if time_sol == 0.0:
                    t_match = re.search(
                        r"Time\s+\(sec\):\s+([0-9\.eE\+\-]+)", log_content
                    )
                    if t_match:
                        time_sol = float(t_match.group(1))
            except Exception:
                pass

        # Only append if we actually extracted meaningful data
        if dt > 0 or time_sol > 0 or worked:
            results.append(
                {
                    "n_elems": n_elems,
                    "p_deg": p_deg,
                    "scheme": scheme,
                    "dt": dt,
                    "worked": worked,
                    "time": time_sol,
                    "dir": str(current_dir),
                }
            )

    # Sort results for readability: N -> P -> Scheme -> dt
    results.sort(key=lambda x: (x["n_elems"], x["p_deg"], x["scheme"], x["dt"]))

    header = f"{'N':<5} | {'P':<3} | {'Scheme':<8} | {'dt':<10} | {'Worked':<6} | {'Time (s)':<10} | Directory "
    print(header)
    print("-" * 80)

    # Print the table rows
    missing_times = 0
    for r in results:
        worked_str = "Yes" if r["worked"] else "No"
        dt_str = f"{r['dt']:.2e}" if r["dt"] > 0 else "Unknown"
        time_str = f"{r['time']:.4f}" if r["time"] > 0 else "-"

        if r["time"] == 0.0:
            missing_times += 1

        # if r["worked"]:
        print(
            f"{r['n_elems']:<5} | {r['p_deg']:<3} | {r['scheme']:<8} | {dt_str:<10} | {worked_str:<6} | {time_str:<10} | {r["dir"]}"
        )

    print("-" * 80)
    print(f"Total valid runs found: {len(results)}")
    if missing_times > 0:
        print(
            f"Note: {missing_times} runs lacked a time reading (likely crashed early)."
        )
    print("\n")


if __name__ == "__main__":
    """
    grep -e '"did_end": true' build/circ_expl/*/*/*/*.json
    """
    summarize_tree_results(base_dir="build/circ_expl_gamma98/", target_t_f=0.13)
    summarize_tree_results(base_dir="build/circ_expl_mixed/", target_t_f=0.13)
    summarize_tree_results(base_dir="build/circ_expl/", target_t_f=0.13)
    exit()
