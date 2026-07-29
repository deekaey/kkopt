# kkopt_postprocess_sobol.py

def _build_salib_problem_from_project(project):
    names = []
    bounds = []
    for k, v in project.setting.parameters.items():
        if v['distribution'] != 'uniform':
            raise NotImplementedError(
                f"SALib wrapper supports only uniform parameters, got {v['distribution']} for {k}"
            )
        names.append(k)
        bounds.append([v['minvalue'], v['maxvalue']])
    return {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds,
    }

def salib_sobol_analysis_from_y(project):
    """
    Compute Sobol indices from stored param_values and Y (no simulation).
    Expects:
      <output>_sobol_N<reps>_params.npy
      <output>_sobol_N<reps>_Y.csv
    and writes:
      <output>_sobol_N<reps>_S1.csv
      <output>_sobol_N<reps>_ST.csv
      <output>_sobol_N<reps>_S2.csv
    """
    suffix = _rep_suffix(project)
    base = project.setting.output + "_sobol" + suffix
    out_dir = project.output_dir
    os.makedirs(out_dir, exist_ok=True)

    param_file = base + "_params.npy"
    y_file = base + "_Y.csv"

    if not os.path.exists(param_file) or not os.path.exists(y_file):
        print(
            f"[salib_sobol_analysis_from_y] Missing files: "
            f"{param_file} or {y_file}; cannot analyze."
        )
        return

    param_values = np.load(param_file)
    df_y = pd.read_csv(y_file)
    Y = df_y["Y"].to_numpy()

    if param_values.shape[0] != Y.shape[0]:
        print(
            f"[salib_*_analysis_from_y] Mismatch between param_values ({param_values.shape[0]}) "
            f"and Y ({Y.shape[0]}) after NaN removal"
        )
        return

    valid_idx = ~np.isnan(Y)
    if not np.all(valid_idx):
        print(f"SALib: {np.sum(~valid_idx)} runs had NaN output, excluding them")
        param_values = param_values[valid_idx, :]
        Y = Y[valid_idx]

    problem = _build_salib_problem_from_project(project)
    Si = sobol.analyze(problem, Y, print_to_console=True)

    np.savetxt(
        base + "_S1.csv",
        np.vstack([problem['names'], Si['S1']]).T,
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        base + "_ST.csv",
        np.vstack([problem['names'], Si['ST']]).T,
        delimiter=",",
        fmt="%s",
    )

    if 'S2' in Si:
        D = problem['num_vars']
        names = problem['names']
        S2_list = []
        for i in range(D):
            for j in range(i + 1, D):
                S2_list.append([names[i], names[j], Si['S2'][i, j]])
        S2_arr = np.array(S2_list, dtype=object)
        np.savetxt(
            base + "_S2.csv",
            S2_arr,
            delimiter=",",
            fmt="%s",
        )

def salib_morris_analysis_from_y(project):
    """
    Compute Morris indices from stored param_values and Y (no simulation).

    Expects:
      <output>_morris_N<reps>_params.npy
      <output>_morris_N<reps>_Y.csv

    and writes:
      <output>_morris_N<reps>_indices.csv
    """
    suffix = _rep_suffix(project)
    base = project.setting.output + "_morris" + suffix
    out_dir = project.output_dir
    os.makedirs(out_dir, exist_ok=True)

    param_file = base + "_params.npy"
    y_file = base + "_Y.csv"

    if not os.path.exists(param_file) or not os.path.exists(y_file):
        print(
            f"[salib_morris_analysis_from_y] Missing files: "
            f"{param_file} or {y_file}; cannot analyze."
        )
        return

    # Load data
    param_values = np.load(param_file)
    df_y = pd.read_csv(y_file)
    Y = df_y["Y"].to_numpy()

    problem = _build_salib_problem_from_project(project)
    D = problem["num_vars"]

    # ----- Check basic consistency -----
    if param_values.shape[0] != Y.shape[0]:
        print(
            f"[salib_morris_analysis_from_y] ERROR: param_values (n={param_values.shape[0]}) "
            f"and Y (n={Y.shape[0]}) lengths differ."
        )
        return

    denom = D + 1
    if param_values.shape[0] % denom != 0:
        print(
            f"[salib_morris_analysis_from_y] ERROR: total runs {param_values.shape[0]} "
            f"is not a multiple of (D+1) = {denom}. Cannot form full trajectories."
        )
        return

    k = param_values.shape[0] // denom  # number of trajectories
    print(f"[salib_morris_analysis_from_y] Detected D={D}, k={k}, total runs={param_values.shape[0]}")

    # ----- Reshape into (traj, step, var) and (traj, step) -----
    # param_values: (k*(D+1), D) -> (k, D+1, D)
    param_values_3d = param_values.reshape(k, denom, D)
    # Y: (k*(D+1),) -> (k, D+1)
    Y_2d = Y.reshape(k, denom)

    # ----- Identify valid trajectories -----
    # A trajectory is valid if none of its Y entries is NaN
    traj_valid = ~np.isnan(Y_2d).any(axis=1)
    n_invalid = np.sum(~traj_valid)

    if n_invalid > 0:
        print(
            f"[salib_morris_analysis_from_y] {n_invalid} out of {k} trajectories "
            f"contain NaNs and will be discarded."
        )

    if np.sum(traj_valid) == 0:
        print("[salib_morris_analysis_from_y] ERROR: No valid trajectories left after discarding NaNs.")
        return

    # Keep only valid trajectories
    param_values_3d_valid = param_values_3d[traj_valid, :, :]   # (k_valid, D+1, D)
    Y_2d_valid = Y_2d[traj_valid, :]                            # (k_valid, D+1)

    # Flatten back to (n_valid_runs, D) and (n_valid_runs,)
    param_values_valid = param_values_3d_valid.reshape(-1, D)
    Y_valid = Y_2d_valid.reshape(-1)

    print(
        f"[salib_morris_analysis_from_y] Using {param_values_valid.shape[0]} runs "
        f"from {param_values_3d_valid.shape[0]} valid trajectories."
    )

    # ----- Run Morris analysis with valid data -----
    Si = morris_analyze.analyze(
        problem,
        param_values_valid,
        Y_valid,
        print_to_console=True
    )

    # Save indices
    arr = np.vstack([
        problem['names'],
        Si['mu_star'],
        Si['sigma'],
        Si['mu'],
    ]).T
    header = "name,mu_star,sigma,mu"
    np.savetxt(
        base + "_indices.csv",
        arr,
        delimiter=",",
        fmt="%s",
        header=header,
        comments="",
    )


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
