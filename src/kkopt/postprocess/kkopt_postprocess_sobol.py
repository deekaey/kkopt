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
