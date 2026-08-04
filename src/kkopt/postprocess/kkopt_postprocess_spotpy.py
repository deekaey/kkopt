# kkopt_postprocess.py
import os
import math
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotpy

from kkopt.postprocess.common import _rep_suffix, indent, rmse, rrmse, r2

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

from collections import defaultdict
from spotpy import objectivefunctions as obj

import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import xml.etree.ElementTree as ET


def plot_parameter_correlations(
    project,
    df_used,
    param_cols,
    method,
    like_type):

    # clean parameter names
    df_params = df_used[param_cols].copy()
    clean_names = [
        c[3:] if c.startswith("par") else c
        for c in param_cols
    ]
    df_params.columns = clean_names

    # correlation matrix
    corr = df_params.corr(method="pearson")

    # heatmap plot
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


def sir_resample_indices_from_sse(
    SSE,
    beta=1.0,
    sigma2=None,
    method="stratified",
    n_resample=None,
    rng=None):

    SSE = np.asarray(SSE, dtype=float)
    N = SSE.size
    if N == 0:
        return np.array([], dtype=int), np.array([]), np.nan
    if n_resample is None:
        n_resample = N
    if rng is None:
        rng = np.random.default_rng()
    # sigma2 (heuristic assumption)
    if sigma2 is None:
        k = max(1, int(0.1 * N))
        sigma2 = max(1e-12, np.mean(np.sort(SSE)[:k]) / 2.0)
    logw = -SSE / (2.0 * sigma2)
    logw -= np.max(logw)
    w = np.exp(logw)
    if beta != 1.0:
        w = w ** beta
    w = np.where(np.isfinite(w), w, 0.0)
    w_sum = w.sum()
    p = np.ones(N) / N if w_sum <= 0 else w / w_sum
    # resampling
    if method == "systematic":
        positions = (rng.random() + np.arange(n_resample)) / n_resample
        cumsum = np.cumsum(p)
        idx = np.zeros(n_resample, dtype=int)
        i = j = 0
        while i < n_resample:
            if positions[i] < cumsum[j]:
                idx[i] = j; i += 1
            else:
                j += 1
    elif method == "stratified":
        u = (np.arange(n_resample) + rng.random(n_resample)) / n_resample
        cumsum = np.cumsum(p)
        idx = np.searchsorted(cumsum, u, side="right")
    else:
        idx = rng.choice(N, size=n_resample, replace=True, p=p)
    ESS = 1.0 / np.sum(p ** 2)
    return idx, p, ESS


def plot_parameter_distribution_resampled(
    project,
    df_resampled,
    param_cols,
    method_lower,
    like_type,
    suffix,
    tag="sir"):

    cols_per_row = 5
    n_params = len(param_cols)
    if n_params == 0:
        print("[plot_param_dist_resampled] No parameter columns.")
        return
    n_rows = math.ceil(n_params / cols_per_row)
    fig_w = 3.2 * cols_per_row
    fig_h = 2.8 * n_rows
    plt.figure(figsize=(fig_w, fig_h))
    for i, param in enumerate(param_cols):
        ax = plt.subplot(n_rows, cols_per_row, i + 1)
        if param in df_resampled.columns:
            data = pd.to_numeric(df_resampled[param], errors="coerce")
            sns.histplot(data, kde=True, ax=ax)
        else:
            ax.text(0.5, 0.5, f"{param} missing", ha="center", va="center")
        display_name = param[3:] if param.startswith("par") else param
        ax.set_xlabel(display_name)
        ax.set_title("")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle(f"Parameter distributions (SIR resampled, {method_lower}, {like_type})", y=0.98)
    out = os.path.join(project.output_dir, f"{project.setting.output}{suffix}_parameters_{like_type}_{method_lower}_{tag}.png")
    plt.savefig(out, dpi=300)
    plt.close()

    out_csv = os.path.join(project.output_dir, f"{project.setting.output}{suffix}_sir_resampled_params.csv")
    df_resampled[param_cols].to_csv(out_csv, index=False)


def plot_calibration_results(
    project,
    df_sorted,
    output_dir,
    like_type,
    calibration_id_slices,
    sim_cols,
    observed_values,
    param_cols=None,
    best_index=0,
    title=None,
    max_cols=3,
    dpi=150,
    use_sir=False,
    sir_beta=1.0,
    sir_method="stratified",
    sir_frac=0.1,
    rng_seed=None):

    os.makedirs(output_dir, exist_ok=True)
    if len(df_sorted) == 0:
        print("[plot_calibration_results] No data.")
        return

    best = df_sorted.iloc[best_index]
    sim_best = best[sim_cols].to_numpy(dtype=float)
    obs = np.asarray(observed_values, dtype=float)
    if sim_best.shape[0] != obs.shape[0]:
        print("[plot_calibration_results] sim/obs length mismatch.")
        return

    reps = getattr(project.setting, "repetitions", None)
    suffix = f"_N{reps}" if reps is not None else ""

    # layout
    n_ids = len(calibration_id_slices)
    n_cols = min(max_cols, n_ids) if n_ids > 0 else 1
    n_rows_scatter = ceil(n_ids / n_cols)
    total_rows = n_rows_scatter + 1  # +1 for table

    fig = plt.figure(figsize=(4.0 * n_cols, 3.2 * total_rows), dpi=dpi)
    gs = fig.add_gridspec(total_rows, n_cols, height_ratios=[1] * n_rows_scatter + [0.9])

    cids = list(calibration_id_slices.keys())

    # SIR
    idx_resampled = None
    if use_sir:
        if "SSE" in df_sorted.columns or "SSE_norm" in df_sorted.columns:
            SSE_col = "SSE_norm" if "SSE_norm" in df_sorted.columns else "SSE"
            N = len(df_sorted)
            n_resample = max(1, min(N, int(sir_frac * N)))
            rng = np.random.default_rng(rng_seed)
            # Erwartet externe Funktion sir_resample_indices_from_sse
            idx_resampled, p, ess = sir_resample_indices_from_sse(
                SSE=df_sorted[SSE_col].to_numpy(dtype=float),
                beta=sir_beta,
                method=sir_method,
                n_resample=n_resample,
                rng=rng)
        else:
            print("[plot_calibration_results] No SSE/SSE_norm column. Skipping SIR overlay.")
            use_sir = False

    # Scatter-Subplots
    for i, cid in enumerate(cids):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(gs[r, c])

        s, e = calibration_id_slices[cid]
        obs_c = obs[s:e]
        sim_c_best = sim_best[s:e]

        # SIR
        if use_sir and idx_resampled is not None and len(idx_resampled) > 0:
            sims_block = df_sorted.iloc[idx_resampled][sim_cols].to_numpy(dtype=float)  # (n_resample, total_points)
            sims_c = sims_block[:, s:e]
            order = np.argsort(obs_c)
            obs_sorted = obs_c[order]
            q50 = np.nanmedian(sims_c, axis=0)[order]
            q05 = np.nanpercentile(sims_c, 5, axis=0)[order]
            q95 = np.nanpercentile(sims_c, 95, axis=0)[order]
            ax.fill_between(obs_sorted, q05, q95, color="tab:blue", alpha=0.15, label="SIR 5–95%")
            ax.plot(obs_sorted, q50, color="tab:blue", alpha=0.8, lw=1.2, label="SIR Median")

        # best
        ax.scatter(obs_c, sim_c_best, alpha=0.7, edgecolor="none", color="tab:orange", label="Best run")

        # 1:1 Linie
        vmin = np.nanmin([np.nanmin(obs_c), np.nanmin(sim_c_best)])
        vmax = np.nanmax([np.nanmax(obs_c), np.nanmax(sim_c_best)])
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Simulated")
        ax.set_title(f"{cid} - best by {like_type}")

        metric_col = f"{like_type}_{cid}"
        if metric_col in df_sorted.columns:
            val = best[metric_col]
            try:
                txt = f"{like_type}: {val:.4g}"
            except Exception:
                txt = f"{like_type}: {val}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8))

        if use_sir and idx_resampled is not None and len(idx_resampled) > 0:
            ax.legend(loc="best", fontsize=8, framealpha=0.8)

    # empty cells
    for j in range(n_ids, n_rows_scatter * n_cols):
        r = j // n_cols
        c = j % n_cols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    # table
    ax_tbl = fig.add_subplot(gs[-1, :])
    ax_tbl.axis("off")

    rows = []
    if param_cols:
        for p in param_cols:
            if p in best.index:
                rows.append((p, best[p]))
    if "score" in best.index:
        rows.append(("score", best["score"]))
    per_id_cols = [c for c in df_sorted.columns if c.startswith(f"{like_type}_")]
    for col in per_id_cols:
        rows.append((col, best[col]))

    if rows:
        def fmt(v):
            try:
                if pd.isna(v):
                    return "nan"
                if isinstance(v, (int, np.integer)):
                    return f"{int(v)}"
                if isinstance(v, (float, np.floating)):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return str(v)
        cell_text = [[fmt(n), fmt(w)] for n, w in rows]
        table = ax_tbl.table(cellText=cell_text,
                             colLabels=["Name", "Wert"],
                             loc="center",
                             cellLoc="left",
                             colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)
    else:
        ax_tbl.text(0.5, 0.5, "No parameters/metrics available",
                    ha="center", va="center", fontsize=10)

    if title is None:
        title = f"Calibration results - best by {like_type}"
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(output_dir, f"{project.setting.output}{suffix}_best_result_on_{like_type}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # export parameters
    best_params = {}
    if param_cols:
        for p in param_cols:
            if p in best.index:
                best_params[p] = best[p]

    try:
        root = ET.Element("parameters")
        for name, value in best_params.items():
            ET.SubElement(
                root,
                "par",
                name=str(name).replace("par", ""),
                value=str(value),)

        indent(root)
        tree = ET.ElementTree(root)
        xml_path = f"{project.setting.output}{suffix}_best_parameters.xml"
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    except Exception as e:
        print(f"[plot_calibration_results] Could not write best parameters XML: {e}")


def spotpy_postprocess(project, method="mcmc"):

    os.makedirs(project.output_dir, exist_ok=True)

    # 1) base file
    reps = getattr(project.setting, "repetitions", None)
    suffix = f"_N{reps}" if reps is not None else ""
    base_file = f"{project.setting.output}{suffix}_base.csv"
    base = pd.read_csv(base_file)
    for col in ("datetime", "time"):
        if col in base.columns:
            base = base.drop(columns=[col])
    required_base_cols = {"calibration_id", "evaluation", "simulation"}
    missing = required_base_cols - set(base.columns)
    if missing:
        raise ValueError(f"Base file missing columns: {missing}")
    base = base.reset_index(drop=True)

    observed_values = base["evaluation"].to_numpy()
    # calibration_id -> Slices
    id_groups = base.groupby("calibration_id", sort=False).size()
    id_slices = {}
    start = 0
    for cid, size in id_groups.items():
        id_slices[cid] = (start, start + size)
        start += size
    cal_ids = list(id_slices.keys())

    # 2) load all results
    df_file = f"{project.setting.output}{suffix}.csv"
    df = pd.read_csv(df_file, delimiter=",")
    param_cols = [c for c in df.columns if c.startswith("par")]
    sim_cols = [c for c in df.columns if c.startswith("simulation")]
    if len(sim_cols) != len(observed_values):
        raise ValueError(
            f"Mismatch between sim_cols ({len(sim_cols)}) and observed length ({len(observed_values)}). "
            "Ensure sim_cols ordering matches base.evaluation ordering.")

    sims = df[sim_cols].to_numpy(dtype=float)   # (n_runs, n_points)
    obs = observed_values.astype(float)         # (n_points,)

    # 3) normalized sum of squared errors SSE per ID
    errors2_sum = np.zeros(sims.shape[0], dtype=float)
    for cid, (s, e) in id_slices.items():
        obs_c = obs[s:e]
        sims_c = sims[:, s:e]
        scale = np.nanmean(np.abs(obs_c))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        err = (sims_c - obs_c[None, :]) / scale
        errors2_sum += np.nansum(err ** 2, axis=1)
    df["SSE_norm"] = errors2_sum

    # 4) score per id
    like_type = project.setting.likelihood
    per_id_metrics = {}
    for cid in cal_ids:
        s, e = id_slices[cid]
        obs_c = obs[s:e]
        sims_c = sims[:, s:e]
        if like_type == "r2":
            per_id_metrics[cid] = r2(sims_c, obs_c)
        elif like_type == "rmse":
            per_id_metrics[cid] = rmse(sims_c, obs_c)
        elif like_type == "rrmse":
            per_id_metrics[cid] = rrmse(sims_c, obs_c)
        else:
            raise ValueError(f"Unknown output metric: {like_type}")

    per_id_df = pd.DataFrame( per_id_metrics)  # n_runs x n_ids
    df = pd.concat([df, per_id_df.add_prefix(f"{like_type}_")], axis=1)

    # mean global score
    df["score"] = per_id_df.mean(axis=1, skipna=True)
    ascending = (like_type != "r2")

    # 5)
    df_chain = df.copy()
    df_sorted_all = df.sort_values(by="score", ascending=ascending).reset_index(drop=True)

    # 6) subsets depending on method
    method_lower = method.lower()
    percentile_threshold = None
    if method_lower == "mcmc":
        burnin_frac = 0.5
        burnin = int(len(df_chain) * burnin_frac)
        df_used = df_chain.iloc[burnin:].reset_index(drop=True)
        df_sorted_used = df_used.sort_values(by="score", ascending=ascending).reset_index(drop=True)
    else:
        percentile_threshold = 0.01  # Top 1%
        top_n = max(1, int(len(df_sorted_all) * percentile_threshold))
        df_used = df_sorted_all.head(top_n).reset_index(drop=True)
        df_sorted_used = df_used  # already sorted

    if len(df_used) == 0:
        print(f"[spotpy_postprocess] No usable samples found for method={method}.")
        return

    # 7) parameter plots (df_used)
    n_params = len(param_cols)
    if (n_params > 0) and len(df_used) > 0:
        plot_parameter_correlations(project, df_used, param_cols, method_lower, like_type)
        plot_parameter_distribution(project, df_used, param_cols, method_lower, like_type, percentile_threshold, suffix)

    # 8) LHS/FAST: SIR-Resampling
    if n_params > 0 and len(df_sorted_all) > 0 and method_lower in ("lhs", "fast"):
        N = len(df_sorted_all)
        n_resample = min(N, max(200, int(0.25 * N)))
        idx_sir, p, ess = sir_resample_indices_from_sse(
            SSE=df_sorted_all["SSE_norm"].to_numpy(dtype=float),
            beta=1.0,
            method="stratified",
            n_resample=n_resample,
            rng=np.random.default_rng(42),
        )
        if idx_sir.size > 0:
            df_resampled = df_sorted_all.iloc[idx_sir].reset_index(drop=True)
            plot_parameter_distribution_resampled(project, df_resampled, param_cols, method_lower, like_type, suffix, tag="sir")
            # optional CSV
            if False:
                try:
                    out_csv = os.path.join(project.output_dir, f"{project.setting.output}{suffix}_sir_resampled_params.csv")
                    df_resampled[param_cols].to_csv(out_csv, index=False)
                except Exception as e:
                    print(f"[spotpy_postprocess] Could not save SIR resampled params CSV: {e}")

    # 9) plot best simulation
    if method_lower == "mcmc":
        df_for_best = df_sorted_used   # post-burn-in, sorted
    else:
        df_for_best = df_sorted_all    # all, sorted

    plot_calibration_results(
        project=project,
        df_sorted=df_for_best,
        output_dir=project.output_dir,
        like_type=like_type,
        calibration_id_slices=id_slices,
        sim_cols=sim_cols,
        observed_values=observed_values,
        param_cols=param_cols,
        best_index=0,
        title=None,
        max_cols=3,
        dpi=150)
