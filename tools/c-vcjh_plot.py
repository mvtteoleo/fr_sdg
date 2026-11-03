import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_parse_json(filepath):
    """Parses the JSON file into a pandas DataFrame."""
    with open(filepath, "r") as f:
        data = json.load(f)

    max_t = data["case"]["max_t"]
    records = []

    for item in data["case"]["ris"]:
        inp = item["inp"]
        out = item["ris"]

        # Determine if the run successfully reached the end time without blowing up
        err_list = out.get("err", [None])
        err_rho = err_list[0] if err_list and err_list[0] is not None else None
        max_time = out.get("max_time")
        success = (max_time == max_t) and (err_rho is not None)

        records.append(
            {
                "p": inp.get("p"),
                "cfl": inp.get("cfl"),
                "c_key": inp.get("c_key"),
                "c_val": inp.get("c_val"),
                "time_to_solution": out.get("time_to_solution"),
                "err_rho": err_rho,
                "success": success,
            }
        )

    return pd.DataFrame(records)


def plot_stability_ceiling(df):
    """Plots the Maximum CFL (Stability Ceiling) vs c_key using pure Matplotlib."""
    # Filter only successful runs and find the max CFL for each (p, c_key) combination
    successful_runs = df[df["success"] == True]
    cfl_max_df = successful_runs.groupby(["p", "c_key"])["cfl"].max().reset_index()

    c_order = ["cDG", "cSD", "cHU", "c_"]
    unique_ps = sorted(cfl_max_df["p"].unique())

    # Setup for a manual grouped bar chart
    x = np.arange(len(c_order))
    width = 0.8 / len(unique_ps)

    plt.figure(figsize=(8, 5))

    for i, p in enumerate(unique_ps):
        p_data = cfl_max_df[cfl_max_df["p"] == p]

        # Match the y-values to the specific c_order
        y_vals = []
        for c in c_order:
            val = p_data[p_data["c_key"] == c]["cfl"]
            y_vals.append(val.values[0] if not val.empty else 0)

        # Calculate the offset for the grouped bars
        offset = (i - len(unique_ps) / 2 + 0.5) * width
        plt.bar(x + offset, y_vals, width, label=f"p={p}")

    plt.title("Stability: Maximum Stable CFL vs $c_{key}$")
    plt.xlabel("c VCJH Type")
    plt.ylabel("Max Stable CFL")
    plt.xticks(x, c_order)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Poly Deg (p)")
    plt.tight_layout()
    plt.savefig("stability_cfl_max.png", dpi=300)
    plt.show()


def plot_accuracy_at_subceiling(df, target_cfl=0.01):
    """Plots the L2 Error vs c_key using pure Matplotlib."""
    sub_df = df[(df["cfl"] == target_cfl) & (df["success"] == True)]

    c_order = ["cDG", "cSD", "cHU", "c_"]
    unique_ps = sorted(sub_df["p"].unique())

    plt.figure(figsize=(8, 5))

    for p in unique_ps:
        p_data = sub_df[sub_df["p"] == p]

        # Match data to the requested c_order
        x_vals = []
        y_vals = []
        for c in c_order:
            val = p_data[p_data["c_key"] == c]["err_rho"]
            if not val.empty:
                x_vals.append(c)
                y_vals.append(val.values[0])

        plt.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=8, label=f"p={p}")

    plt.yscale("log")
    plt.title(f"Accuracy: $L^2$ Error vs $c_{{key}}$ (Fixed CFL = {target_cfl})")
    plt.xlabel("c VCJH Type")
    plt.ylabel(r"Density $L^2$ Error (Log Scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Poly Deg (p)")
    plt.tight_layout()
    plt.savefig("accuracy_tradeoff.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # df = load_and_parse_json("outputs/c-vcjh-results.json")
    df = load_and_parse_json("outputs/after_check.json")
    plot_stability_ceiling(df)
    plot_accuracy_at_subceiling(df, target_cfl=0.01)
