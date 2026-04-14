# kkopt_postprocess.py
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotpy


def _rep_suffix(project) -> str:
    """
    Return a suffix encoding the number of repetitions, e.g. '_N6000'.
    Falls back to empty string if repetitions is missing.
    """
    reps = getattr(project.setting, "repetitions", None)
    if reps is None:
        return ""
    try:
        n = int(reps)
    except Exception:
        return ""
    return f"_N{n}"


def postprocess(project):
    """
    Dispatch to the appropriate postprocessing depending on the method.

    - SpotPy calibration methods ('mcmc', 'fast', 'lhs') → spotpy_postprocess
    - SALib Sobol ('sobol') → salib_sobol_postprocess
    - SALib Morris ('morris') → salib_morris_postprocess
    """
    method = getattr(project.setting, "method", "").lower()

    if method in ["mcmc", "fast", "lhs"]:
        spotpy_postprocess(project)
    elif method == "sobol":
        salib_sobol_postprocess(project)
    elif method == "morris":
        salib_morris_postprocess(project)
    else:
        print(f"[postprocess] No postprocessing implemented for method='{method}'")


# -------------------------------------------------------------------------
# SpotPy calibration postprocessing
# -------------------------------------------------------------------------
def spotpy_postprocess(project):
    base = pd.read_csv(f"{project.setting.output}_base.csv")
    base = base.set_index(pd.to_datetime(base.datetime))

    percentile_threshold = 0.2
    delimiter = ","
    observed_values = base["evaluation"]
    like_type = "RMSE"  # or 'R2'

    df = pd.read_csv(project.output_file, delimiter=delimiter)

    like_col = "like1"
    param_cols = [col for col in df.columns if col.startswith("par")]
    sim_cols = [col for col in df.columns if col.startswith("simulation_")]

    if like_type == "R2":
        df["R2"] = df[sim_cols].apply(
            lambda row: spotpy.objectivefunctions.rsquared(
                row.values, observed_values
            ),
            axis=1,
        )
        df_sorted = df.sort_values(by=like_col, ascending=False)
        top_n = int(len(df_sorted) * percentile_threshold)
        df_top = df_sorted.head(top_n)
    else:
        df["RMSE"] = df[sim_cols].apply(
            lambda row: spotpy.objectivefunctions.rmse(row.values, observed_values),
            axis=1,
        )
        df_sorted = df.sort_values(by="RMSE", ascending=True)
        top_n = int(len(df_sorted) * percentile_threshold)
        df_top = df_sorted.head(top_n)

    os.makedirs(project.output_dir, exist_ok=True)

    # === PARAMETER DISTRIBUTIONS (top X%) ===
    n_params = len(param_cols)
    if n_params > 0:
        cols_per_row = 5
        n_rows = math.ceil(n_params / cols_per_row)
        plt.figure(figsize=(cols_per_row * 3, n_rows * 3))

        for i, param in enumerate(param_cols):
            plt.subplot(n_rows, cols_per_row, i + 1)
            sns.histplot(df_top[param], kde=True)
            # lines for the best 3 runs
            for j in range(min(3, len(df_sorted))):
                best_params = df_sorted.iloc[j]
                plt.axvline(
                    best_params[param],
                    color="gold",
                    linestyle="--",
                    linewidth=1.5,
                )
            # initial value from configuration
            pname = param[3:]  # 'parX' -> parameter name
            if pname in project.setting.parameters:
                init_val = project.setting.parameters[pname]["initialvalue"]
                plt.axvline(
                    init_val,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                )
            plt.title(param)

        plt.tight_layout()
        plt.suptitle(
            f"Parameterverteilungen (Top {int(percentile_threshold * 100)}%)",
            y=1.02,
        )
        suffix = _rep_suffix(project)
        param_plot_path = os.path.join(
            project.output_dir,
            f"{project.setting.output}{suffix}_parameters_{like_type}.png",
        )
        plt.savefig(param_plot_path, dpi=300)
        plt.close()

    # === BEST SIMULATION PLOT WITH TABLE ===
    if len(df_top) == 0 or len(sim_cols) == 0:
        print("[spotpy_postprocess] No simulations found to plot.")
        return

    sim_array = df_top[sim_cols].to_numpy()
    best_sim = df_sorted.iloc[0][sim_cols].to_numpy()

    lower = np.percentile(sim_array, 5, axis=0)
    upper = np.percentile(sim_array, 95, axis=0)
    error = [
        np.maximum(0.0, best_sim - lower),
        np.maximum(0.0, upper - best_sim),
    ]

    best_like = df_sorted.iloc[0][like_type]
    min_val = min(observed_values.min(), best_sim.min())
    max_val = max(observed_values.max(), best_sim.max())

    import matplotlib.gridspec as gridspec

    best_params = df_sorted.iloc[0][param_cols]

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # left: scatter with uncertainty
    ax0 = fig.add_subplot(gs[0])
    ax0.errorbar(
        observed_values,
        best_sim,
        yerr=error,
        fmt="o",
        ecolor="lightblue",
        alpha=0.6,
        label=f"Unsicherheitsband (Top {int(percentile_threshold * 100)}%)",
    )
    ax0.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Linie")
    ax0.scatter(
        observed_values,
        best_sim,
        color="blue",
        label=f"Beste Simulation ({like_type} = {best_like:.3f})",
    )
    ax0.set_xlabel("Beobachtete Werte")
    ax0.set_ylabel("Simulierte Werte")
    ax0.set_title("Beste Simulation mit Unsicherheitsband")
    ax0.set_xlim(0, 1.1 * max_val)
    ax0.set_ylim(0, 1.1 * max_val)
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend()

    # right: parameter table
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")

    table_data = [[param, f"{value:.4g}"] for param, value in best_params.items()]
    table = ax1.table(
        cellText=table_data,
        colLabels=["Parameter", "Wert"],
        loc="center",
        cellLoc="left",
    )
    table.scale(3, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if col == 1:
            cell.set_width(0.5)
            cell.set_text_props(ha="left", va="center")

    suffix = _rep_suffix(project)
    scatter_plot_path = os.path.join(
        project.output_dir,
        f"{project.setting.output}{suffix}_opt_{like_type}_with_table.png",
    )
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# SALib / Sobol postprocessing
# -------------------------------------------------------------------------
def salib_sobol_postprocess(project):
    """
    Postprocess Sobol sensitivity results created by run_sensitivity(method='sobol').

    Expects files:
      <output>_sobol_S1.csv
      <output>_sobol_ST.csv
      <output>_sobol_S2.csv
    """
    suffix = _rep_suffix(project)
    base = project.setting.output + "_sobol" + suffix
    out_dir = project.output_dir
    os.makedirs(out_dir, exist_ok=True)

    S1_file = base + "_S1.csv"
    ST_file = base + "_ST.csv"
    S2_file = base + "_S2.csv"

    if not (os.path.exists(S1_file) and os.path.exists(ST_file)):
        print(
            f"[salib_sobol_postprocess] Sobol files not found "
            f"({S1_file}, {ST_file}). Skipping."
        )
        return

    # --- Load S1 and ST ---
    df_S1 = pd.read_csv(S1_file, header=None, names=["name", "S1"])
    df_ST = pd.read_csv(ST_file, header=None, names=["name", "ST"])

    df = df_S1.merge(df_ST, on="name")
    df["S1"] = df["S1"].astype(float)
    df["ST"] = df["ST"].astype(float)
    df_sorted = df.sort_values(by="ST", ascending=False)

    # Save merged indices
    df_sorted.to_csv(base + "_S1_ST_sorted.csv", index=False)

    # --- Bar plot of S1 and ST ---
    names = df_sorted["name"].values
    S1 = df_sorted["S1"].values
    ST = df_sorted["ST"].values

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(0.6 * len(names) + 2, 5))
    plt.bar(x - width / 2, S1, width, label="S1 (First-order)")
    plt.bar(x + width / 2, ST, width, label="ST (Total-order)")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Sobol-Index")
    plt.title("Sobol sensitivity indices")
    if len(ST) > 0:
        plt.ylim(0, 1.1 * max(ST.max(), S1.max()))
    plt.legend()
    plt.tight_layout()

    bar_plot_path = os.path.join(
        out_dir, f"{project.setting.output}{suffix}_sobol_S1_ST.png"
    )
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()

    # --- S2 interaction heatmap (if file exists) ---
    if os.path.exists(S2_file):
        df_S2 = pd.read_csv(S2_file, header=None, names=["i", "j", "S2"])

        names_all = sorted(list(set(df_S2["i"]).union(set(df_S2["j"]))))

        def shorten(name: str) -> str:
            if "." in name:
                return name.split(".")[-1]
            return name

        labels = [shorten(n) for n in names_all]

        name_to_idx = {n: i for i, n in enumerate(names_all)}
        mat = np.zeros((len(names_all), len(names_all)))

        for _, row in df_S2.iterrows():
            i = name_to_idx[row["i"]]
            j = name_to_idx[row["j"]]
            val = float(row["S2"])
            mat[i, j] = val
            mat[j, i] = val  # symmetric

        n = len(names_all)
        width = max(6, min(0.4 * n + 2, 16))
        height = width

        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(mat, cmap="viridis", interpolation="nearest")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )
        for tick in ax.get_yticklabels():
            tick.set_fontsize(8)

        ax.set_title("Sobol S2 interaction indices", pad=20)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("S2", rotation=90)
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        heatmap_path = os.path.join(
            out_dir, f"{project.setting.output}_sobol{suffix}_S2.png"
        )
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
    else:
        print(f"[salib_sobol_postprocess] {S2_file} not found. No S2 heatmap created.")


# -------------------------------------------------------------------------
# SALib / Morris postprocessing
# -------------------------------------------------------------------------
def salib_morris_postprocess(project):
    """
    Postprocess Morris sensitivity results created by run_sensitivity(method='morris').

    Expects file:
      <output>_morris_indices.csv
    with columns: name, mu_star, sigma, mu
    """
    suffix = _rep_suffix(project)
    base = project.setting.output + "_morris" + suffix
    out_dir = project.output_dir
    os.makedirs(out_dir, exist_ok=True)

    indices_file = base + "_indices.csv"
    if not os.path.exists(indices_file):
        print(
            f"[salib_morris_postprocess] Morris indices file not found: "
            f"{indices_file}. Skipping."
        )
        return

    df = pd.read_csv(indices_file)
    # ensure numeric
    for col in ["mu_star", "sigma", "mu"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # sort by mu_star descending
    df_sorted = df.sort_values(by="mu_star", ascending=False)

    names = df_sorted["name"].values
    mu_star = df_sorted["mu_star"].values
    sigma = df_sorted["sigma"].values

    # --- Bar plot for mu* and sigma ---
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(0.6 * len(names) + 2, 5))
    plt.bar(x - width / 2, mu_star, width, label="mu* (importance)")
    plt.bar(x + width / 2, sigma, width, label="sigma (nonlinearity/interactions)")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Morris indices")
    plt.title("Morris sensitivity (mu* and sigma)")
    plt.legend()
    plt.tight_layout()

    bar_path = os.path.join(
        out_dir, f"{project.setting.output}_morris{suffix}_mu_sigma_bar.png"
    )
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # --- Scatter plot mu* vs sigma ---
    plt.figure(figsize=(6, 5))
    plt.scatter(mu_star, sigma, c="C0")
    for n, x_val, y_val in zip(names, mu_star, sigma):
        plt.text(x_val, y_val, n, fontsize=8, ha="left", va="bottom")

    plt.xlabel("mu* (mean absolute elementary effect)")
    plt.ylabel("sigma (standard deviation)")
    plt.title("Morris: mu* vs sigma")
    plt.tight_layout()

    scatter_path = os.path.join(
        out_dir, f"{project.setting.output}_morris{suffix}_mu_vs_sigma.png"
    )
    plt.savefig(scatter_path, dpi=300)
    plt.close()
