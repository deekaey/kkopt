# kkopt_postprocess.py
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotpy
from kkopt.postprocess.common import _rep_suffix

import math
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_parameter_correlations(
    project,
    df_used,
    param_cols,
    method,
    like_type):
    # Prepare DataFrame with cleaned parameter names
    df_params = df_used[param_cols].copy()
    clean_names = [
        c[3:] if c.startswith("par") else c
        for c in param_cols
    ]
    df_params.columns = clean_names

    # Compute correlation matrix
    corr = df_params.corr(method="pearson")

    # Plot heatmap
    plt.figure(figsize=(0.5 * len(clean_names) + 4, 0.5 * len(clean_names) + 4))
    sns.heatmap(
        corr,
        xticklabels=clean_names,
        yticklabels=clean_names,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Parameter correlation (MCMC)")
    plt.tight_layout()

    suffix = _rep_suffix(project)
    corr_plot_path = os.path.join(
        project.output_dir,
        f"{project.setting.output}{suffix}_parameters_corr_{method.lower()}.png",
    )
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()


def plot_parameter_distribution(
    project,
    df_used,
    param_cols,
    method_lower,
    like_type,
    percentile_threshold,
    suffix):

    cols_per_row = 5

    n_params = len(param_cols)
    n_rows = math.ceil(n_params / cols_per_row)

    plt.figure(figsize=(cols_per_row * 3, n_rows * 3))

    for i, param in enumerate(param_cols):
        ax = plt.subplot(n_rows, cols_per_row, i + 1)
        sns.histplot(df_used[param], kde=True, ax=ax)

        display_name = param[3:] if param.startswith("par") else param
        ax.set_xlabel(display_name)
        ax.set_title("")

    if method_lower == "mcmc":
        title_text = "Parameter distributions (MCMC, after burn-in)"
    else:
        title_text = f"Parameter distributions (Top {int(percentile_threshold * 100)}%)"

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle(title_text, y=0.98)

    param_plot_path = os.path.join(
        project.output_dir,
        f"{project.setting.output}{suffix}_parameters_{like_type}_{method_lower}.png",
    )
    plt.savefig(param_plot_path, dpi=300)
    plt.close()


def plot_parameter_distribution_resampled(
    project,
    df_sorted: pd.DataFrame,
    param_cols,
    method_lower,
    like_type,
    suffix):

        # LHS/FAST: importance resampling from the full LHS cloud
        # Build weights from like1 (higher = better)
        if "like1" in df_sorted.columns:
            # Extract the 'like1' column as a NumPy array.
            likes = df_sorted["like1"].to_numpy()

            # Small epsilon used to avoid zero weights when we shift likes.
            eps = 1e-12

            # Shift likes so that the smallest like1 becomes ~0.
            # This ensures that all weights are >= eps:
            #   w_i = like_i - min(like) + eps
            # Worst run: like_i = min(like)  -> w_i ≈ eps
            # Best run:  like_i = max(like)  -> w_i ≈ (max - min) + eps
            w = likes - likes.min() + eps

            # Exponent alpha controls how strongly the weights emphasize high-likelihood runs:
            #   - alpha > 1: sharpen weights (good runs get much higher weight)
            #   - alpha = 1: linear weighting (current behavior)
            #   - alpha < 1: flatten weights (differences in like1 matter less)
            alpha = 1.0  # >1 sharpens; <1 flattens
            w = np.power(w, alpha)

            # Sum of weights for normalization.
            w_sum = w.sum()

            if w_sum <= 0:
                # In a pathological case where all weights are zero or NaN
                # (e.g. all likes identical and shifting failed),
                # fall back to uniform sampling across all runs.
                print("[spotpy_postprocess] Degenerate weights, using uniform for resampling.")
                p = np.ones_like(w) / len(w)
            else:
                # Convert weights into probabilities for resampling:
                #   p_i = w_i / Σ_j w_j
                # These probabilities are used as the sampling distribution over the LHS cloud.
                p = w / w_sum

            # Decide how many samples to draw from the weighted LHS cloud.
            # Here: resample 10% of all runs (with replacement).
            # For example, if you had 5000 LHS points, N_resample = 500.
            # You could also cap this with min(1000, len(df_sorted)) if you want a fixed upper limit.
            N_resample = int(0.1 * len(df_sorted))  # number of resampled draws

            # Draw indices from 0..len(df_sorted)-1 with replacement,
            # according to the probability vector p.
            # This is importance resampling: good runs (high like1) are more likely,
            # but all runs (except the very worst) still have a chance to be drawn.
            idx_resampled = np.random.choice(
                len(df_sorted),
                size=N_resample,
                replace=True,
                p=p
            )

            # Build a new DataFrame with the resampled rows.
            # reset_index(drop=True) makes the row index 0..N_resample-1, for convenience.
            df_resampled = df_sorted.iloc[idx_resampled].reset_index(drop=True)


        cols_per_row = 5

        n_params = len(param_cols)
        n_rows = math.ceil(n_params / cols_per_row)

        plt.figure(figsize=(cols_per_row * 3, n_rows * 3))

        for i, param in enumerate(param_cols):
            ax = plt.subplot(n_rows, cols_per_row, i + 1)
            sns.histplot(df_resampled[param], kde=True, ax=ax)

            display_name = param[3:] if param.startswith("par") else param
            ax.set_xlabel(display_name)
            ax.set_title("")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.suptitle(
            "Parameter distributions (LHS/FAST, importance-resampled)",
            y=0.98,
        )

        param_resampled_plot_path = os.path.join(
            project.output_dir,
            f"{project.setting.output}{suffix}_parameters_{like_type}_{method_lower}_resampled.png",
        )
        plt.savefig(param_resampled_plot_path, dpi=300)
        plt.close()


def indent(elem, level=0):
    """Recursively indent XML elements for pretty printing."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def plot_calibration_results(
    project,
    df_sorted: pd.DataFrame,
    output_dir: str = "plots",
) -> None:
    """
    Create one figure containing scatter plots for all calibration IDs
    together with a table of the best parameter values.

    Additionally, export the best parameters as an XML file.
    """

    # ------------------------------------------------------------------
    # Load base file
    # ------------------------------------------------------------------
    reps = getattr(project.setting, "repetitions", None)
    suffix = f"_N{reps}" if reps is not None else ""
    base_file = f"{project.setting.output}{suffix}_base.csv"
    base = pd.read_csv(base_file)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Best simulation (highest objective value)
    # ------------------------------------------------------------------
    best_row = df_sorted.loc[df_sorted["like1"].idxmax()]

    best_params = best_row.filter(like="par")
    best_sim = best_row.filter(like="simulation")

    calib_ids = base["calibration_id"].unique()
    n_scatter = len(calib_ids)

    # ------------------------------------------------------------------
    # Determine subplot layout
    # (+1 panel reserved for parameter table)
    # ------------------------------------------------------------------
    n_panels = n_scatter + 1

    ncols = min(3, math.ceil(math.sqrt(n_panels)))
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    # ------------------------------------------------------------------
    # Scatter plots
    # ------------------------------------------------------------------
    for i, calib_id in enumerate(calib_ids):

        ax = axes[i]

        subset = base[base["calibration_id"] == calib_id]

        observed = subset["evaluation"].to_numpy()

        simulated = np.array(
            [
                best_sim[f"simulation_{idx}"]
                for idx in subset.index
                if f"simulation_{idx}" in best_sim.index
            ]
        )

        ax.scatter(observed, simulated, color="tab:blue", alpha=0.7)

        # Determine axis limits for this calibration only
        xy_min = min(observed.min(), simulated.min())
        xy_max = max(observed.max(), simulated.max())

        pad = 0.05 * (xy_max - xy_min)
        if pad == 0:
            pad = 1e-6  # avoid identical limits

        xy_min -= pad
        xy_max += pad

        ax.plot(
            [xy_min, xy_max],
            [xy_min, xy_max],
            "r--",
            lw=1,
        )

        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"{calib_id}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Simulated")
        ax.grid(True)

    # ------------------------------------------------------------------
    # Parameter table
    # ------------------------------------------------------------------
    table_ax = axes[n_scatter]
    table_ax.axis("off")

    table_data = [
        [name.replace("par", "").lstrip("_"), value]
        for name, value in best_params.items()
    ]

    table = table_ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    table_ax.set_title("Best parameter set")

    # Hide unused axes
    for ax in axes[n_scatter + 1 :]:
        ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        os.path.join(output_dir, "calibration_results.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # Export parameters as XML
    # ------------------------------------------------------------------
    root = ET.Element("parameters")

    for name, value in best_params.items():
        ET.SubElement(
            root,
            "par",
            name=name.replace("par_", ""),
            name=name.replace("par", ""),
            value=str(value),
        )

    tree = ET.ElementTree(root)
    indent(root)
    tree.write(
        os.path.join(output_dir, "best_parameters.xml"),
        encoding="utf-8",
        xml_declaration=True,
    )


def spotpy_postprocess(project, method="mcmc"):

    os.makedirs(project.output_dir, exist_ok=True)

    reps = getattr(project.setting, "repetitions", None)
    suffix = f"_N{reps}" if reps is not None else ""

    # --- 1. Load base output and reconstruct observed_values as stacked evaluation ---
    base_file = f"{project.setting.output}{suffix}_base.csv"
    base = pd.read_csv(base_file)

    # Remove datetime for RMSE computation; we only need values
    if "datetime" in base.columns:
        base = base.drop(columns=["datetime"])

    observed_values = base["evaluation"].to_numpy()

    # --- 2. Load SpotPy output and compute RMSE/R2 vs observed_values ---
    like_type = "RMSE"  # or 'R2'
    delimiter = ","
    df_file = f"{project.setting.output}{suffix}.csv"
    df = pd.read_csv(df_file, delimiter=delimiter)

    like_col = "like1"
    param_cols = [col for col in df.columns if col.startswith("par")]
    sim_cols = [col for col in df.columns if col.startswith("simulation_")]

    if like_type == "R2":
        df = df.copy()  # defragment
        df["R2"] = df[sim_cols].apply(
            lambda row: spotpy.objectivefunctions.rsquared(
                row.values, observed_values
            ),
            axis=1,
        )
        df_sorted = df.sort_values(by=like_col, ascending=False)
    else:
        df = df.copy()  # defragment
        df["RMSE"] = df[sim_cols].apply(
            lambda row: spotpy.objectivefunctions.rmse(row.values, observed_values),
            axis=1,
        )
        df_sorted = df.sort_values(by="RMSE", ascending=True)

    # --- 3. Define subsets depending on method ---
    method_lower = method.lower()

    if method_lower == "mcmc":
        # MCMC: drop burn-in and use all remaining samples
        burnin_frac = 0.5
        burnin = int(len(df_sorted) * burnin_frac)
        df_used = df_sorted.iloc[burnin:].reset_index(drop=True)
    else:
        # LHS/FAST: use top X% best runs for df_used
        percentile_threshold = 0.01  # top x%
        top_n = max(1, int(len(df_sorted) * percentile_threshold))
        df_used = df_sorted.head(top_n).reset_index(drop=True)

    if len(df_used) == 0:
        print(f"[spotpy_postprocess] No usable samples found for method={method}.")
        return


    n_params = len(param_cols)
    if (n_params > 0) and len(df_used) > 0:

        # optional: parameter correlation on post-burn-in samples
        plot_parameter_correlations( project, df_used, param_cols, method_lower, like_type)

        # --- 4. PARAMETER DISTRIBUTIONS (df_used: post-burn-in for MCMC, top 5% for LHS/FAST) ---
        plot_parameter_distribution( project, df_used, param_cols, method_lower, like_type, percentile_threshold, suffix)

    # --- 5. LHS/FAST: Importance-resampled parameter distributions ---
    if (n_params > 0) and len(df_sorted) > 0 and method_lower in ("lhs", "fast") :
        plot_parameter_distribution_resampled( project, df_sorted, param_cols, method_lower, like_type, suffix)

    # --- 6. BEST SIMULATION PLOT WITH TABLE ---
    if len(df_used) == 0 or len(sim_cols) == 0:
        print("[spotpy_postprocess] No simulations found to plot.")
        return

    plot_calibration_results( project, df_sorted, project.output_dir)





def temp( df_used, param_cols, project, method, like_type):
    # Prepare DataFrame with cleaned parameter names
    df_params = df_used[param_cols].copy()
    clean_names = [
        c[3:] if c.startswith("par") else c
        for c in param_cols
    ]
    df_params.columns = clean_names

    # Compute correlation matrix
    corr = df_params.corr(method="pearson")

    # Plot heatmap
    plt.figure(figsize=(0.5 * len(clean_names) + 4, 0.5 * len(clean_names) + 4))
    sns.heatmap(
        corr,
        xticklabels=clean_names,
        yticklabels=clean_names,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Parameter correlation (MCMC)")
    plt.tight_layout()

    suffix = _rep_suffix(project)
    corr_plot_path = os.path.join(
        project.output_dir,
        f"{project.setting.output}{suffix}_parameters_corr_{method.lower()}.png",
    )
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
