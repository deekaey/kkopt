#!/bin/bash
import os
from os.path import exists
import re
import shutil
import subprocess
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import spotpy
from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris as morris_analyze

try:
    from mpi4py import MPI
except ImportError:
    raise Exception("MPI python module mpi4py not available. Exit!")

from kkplot.kkutils.expand import kkexpand
from kkplot.kkutils.log import kklog_debug, kklog_info, kklog_warn
from kkplot.kksources import kkplot_sourcefactory as kkplot_sourcefactory
from kkplot.kkplot_dviplot import *
from kkplot.kkplot_figure import DSSEP

import kkopt.kkutils as utils
from kkopt.kkopt_project import kkopt_project
from kkopt.postprocess.kkopt_postprocess import postprocess

class spot_setup(object):
    def __init__( self, _config, _project):

        self._configuration = _config
        self._setting = _project.setting
        self._parallel = _project.parallel

        # MPI handles (None if not parallel)
        self.comm = MPI.COMM_WORLD if self.parallel else None
        self.rank = self.comm.Get_rank() if self.parallel else 0
        self.size = self.comm.Get_size() if self.parallel else 1

        #self._calibrations = _project.calibrations
        self.likes = []

        #objectivefunction
        self.objective_function = self._setting.get_property( 'likelihood')

        #run test simulation to get test output needed for
        #preparation of evaluation data (maybe not needed or implement on demand)
        #utils.kklog.log_info('Start test simulation')
        self.update_parameters( None)

        self.run_simulation()

        for i, calib in enumerate( self._setting.calibrations):
            # --- SIMULATION variables ---
            sim_cfg = calib.get('simulation', {})
            sim_vars = sim_cfg.get('variables', [])

            for var in sim_vars:
                sim_ds = var['datasource']

                # Adjust provider arguments if present
                if sim_ds.has_provider:
                    args = sim_ds.provider._args
                    new_args = []

                    for j, arg in enumerate(args):
                        arg_expanded = os.path.expandvars(arg)

                        # First argument is usually a script/program; leave it
                        if j == 0:
                            new_args.append(arg)
                            continue

                        # Absolute path -> use rank-specific file selection
                        if arg_expanded.startswith("/"):
                            new_args.append(self._rank_specific_path(arg))
                        # Text/CSV data file -> add rank before extension
                        elif arg_expanded.endswith(".txt") or arg_expanded.endswith(".csv"):
                            new_args.append( self._add_rank_to_path( arg))
                        else:
                            new_args.append(arg)

                    sim_ds.provider._args = new_args

                # Ensure the datasource path itself is rank-specific
                sim_ds.set_path( self._add_rank_to_path( sim_ds.path))

            # --- EVALUATION variables ---
            eval_cfg = calib.get('evaluation', {})
            eval_vars = eval_cfg.get('variables', [])

            for var in eval_vars:
                eval_ds = var['datasource']

                if eval_ds.has_provider:
                    args = eval_ds.provider._args
                    new_args = []

                    for j, arg in enumerate(args):
                        arg_expanded = os.path.expandvars(arg)

                        # Original code modified only argument index 2 for evaluation;
                        # if that is the data file, keep that behavior:
                        if j == 2:
                            new_args.append( self._add_rank_to_path( arg))
                        else:
                            new_args.append(arg)

                    eval_ds.provider._args = new_args

                # Ensure the datasource path itself is rank-specific
                eval_ds.set_path( self._add_rank_to_path( eval_ds.path))

        #prepare evaluation data
        temp = self.get_data( 'simulation')
        self._evaluation = self.get_data( 'evaluation', temp)
        self._simulation = self.get_data( 'simulation', self._evaluation)

        # Only rank 0 writes the base file
        if (not self.parallel) or (self.rank == 0):
            suffix = self._rep_suffix()
            output_path = f"{self._setting.output}{suffix}_base.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            evaluation_df = self.evaluation_df().set_index(['calibration_id', 'time'])
            simulation_df = self.last_simulation_df().set_index(['calibration_id', 'time'])

            df_stacked = pd.DataFrame({
                'evaluation': evaluation_df['value'],
                'simulation': simulation_df['value']
            }).dropna()

            df_stacked.to_csv( output_path)

        self._simulation_default = self._simulation
        self.objectivefunction( self._simulation, self._evaluation)

        #prepare parameters
        self.params = []
        for k,v in self._setting.parameters.items() :
            if v['distribution'] == 'uniform':
                self.params.append( spotpy.parameter.Uniform(
                                                          k,
                                                          v['minvalue'],
                                                          v['maxvalue'],
                                                          v['initialvalue'],
                                                          v['step']))

    def _rep_suffix(self) -> str:
        reps = getattr(self._setting, "repetitions", None)
        if reps is None:
            return ""
        try:
            n = int(reps)
        except Exception:
            return ""
        return f"_N{n}"

    def _get_local_indices( self, n_global: int):
        """Return indices of the global array that this MPI rank should handle."""
        if not self.parallel or self.size == 1:
            return np.arange(n_global)

        base = n_global // self.size
        rest = n_global % self.size

        if self.rank < rest:
            start = self.rank * (base + 1)
            stop = start + base + 1
        else:
            start = rest * (base + 1) + (self.rank - rest) * base
            stop = start + base

        return np.arange(start, stop, dtype=int)

    def _rank_specific_path(self, base_path: str) -> str:
        """
        Given a base file path, build and return a rank-specific variant if in parallel.

        For example, with base:
          .../VN_anlam_soilchemistry-daily.txt

        and rank=1, this method will try (in order):
          VN_anlam_r1soilchemistry-daily.txt     (insert after first "_")
          VN_anlam_soilchemistry-r1daily.txt     (insert after second "_")
          ...

        At each "_" position in the filename, an "_r<rank>" is inserted and
        the existence of the resulting file is checked. The first existing
        file is returned. If none exist, the original base path is returned.
        """
        base_path = os.path.expandvars(base_path)

        if not self.parallel:
            rank = 1
        else:
            rank = self.rank + 1

        dir_name, fname = os.path.split(base_path)

        # Positions of underscores in the filename
        underscore_positions = [i for i, ch in enumerate(fname) if ch == "_"]

        # Try inserting "_r<rank>" after each underscore
        for pos in underscore_positions:
            # Split around this underscore
            before = fname[: pos + 1]   # include the underscore
            after = fname[pos + 1 :]
            rank_fname = f"{before}r{rank}{after}"
            rank_path = os.path.join(dir_name, rank_fname)

            if os.path.exists(rank_path):
                return rank_path

        # Optionally: also try a simple prefix (no underscores present or none matched)
        if not underscore_positions:
            rank_fname = f"r{rank}_{fname}"
            rank_path = os.path.join(dir_name, rank_fname)
            if os.path.exists(rank_path):
                return rank_path

        # If no rank-specific file exists, fall back to base
        kklog_debug(
            f"No rank-specific file found, using base file instead:\n"
            f"  rank={rank}\n  base: {base_path}"
        )
        return base_path

    def _add_rank_to_path(self, path: str) -> str:
        return self._rank_specific_path( path)

    def build_salib_problem( self):
        """
        Build SALib 'problem' dict from kkopt/spotpy parameter configuration.
        Only uniform parameters are supported here.
        """
        names = []
        bounds = []
        for k, v in self._setting.parameters.items():
            if v['distribution'] != 'uniform':
                raise NotImplementedError(
                    f"SALib wrapper currently supports only uniform parameters, got {v['distribution']} for {k}"
                )
            names.append(k)
            bounds.append([v['minvalue'], v['maxvalue']])

        problem = {
            'num_vars': len(names),
            'names': names,
            'bounds': bounds
        }

        return problem

    def run_sensitivity(self, method='sobol', N=1000, output_metric='rmse'):
        """
        Run global sensitivity analysis using SALib.

        Parameters
        ----------
        method : str
            'sobol' or 'morris'.
        N : int
            Target total number of model evaluations (approximate).
        output_metric : str
            'rmse' or 'mean' – how to reduce the time series to one value per run.
        """
        problem = self.build_salib_problem()
        D = problem['num_vars']  # number of parameters
        N_total = int(N)

        # 1) Generate samples
        if method == 'sobol':
            # N_total ≈ N_base * (2D + 2)
            denom = 2 * D + 2
            if denom <= 0:
                raise ValueError(f"Invalid number of parameters D={D} for Sobol")
            N_base = max(1, N_total // denom)
            if self.rank == 0:
                kklog_info(
                    f"[SALib/Sobol] target N_total={N_total}, "
                    f"D={D} -> N_base={N_base}, "
                    f"expected runs ≈ {N_base * denom}"
                )
            param_values = saltelli.sample(problem, N_base, calc_second_order=True)

        elif method == 'morris':
            # N_total ≈ k * (D + 1)
            denom = D + 1
            if denom <= 0:
                raise ValueError(f"Invalid number of parameters D={D} for Morris")
            k = max(1, N_total // denom)
            if self.rank == 0:
                kklog_info(
                    f"[SALib/Morris] target N_total={N_total}, "
                    f"D={D} -> k={k}, "
                    f"expected runs ≈ {k * denom}"
                )
            param_values = morris_sample.sample( problem, N=k, num_levels=4, optimal_trajectories=None)
        else:
            raise ValueError(f"Unknown SALib method: {method}")

        n_runs = param_values.shape[0]
        if self.rank == 0:
            kklog_info(
                f"SALib: generated {n_runs} samples for method={method} "
                f"(target N_total={N_total})"
            )

        if self.parallel:
            n_runs = self.comm.bcast( n_runs, root=0)

        # 2) Distribute parameters to ranks
        if self.parallel:
            local_idx = self._get_local_indices(n_runs)
            kklog_info(
                f"[run_sensitivity] rank={self.rank}, "
                f"local_idx len={len(local_idx)}, "
                f"min={local_idx.min() if len(local_idx) else 'NA'}, "
                f"max={local_idx.max() if len(local_idx) else 'NA'}"
            )
            if self.rank == 0:
                self.comm.bcast( param_values, root=0)
            else:
                param_values = self.comm.bcast(None, root=0)
            local_param_values = param_values[local_idx, :]
        else:
            local_idx = np.arange(n_runs)
            local_param_values = param_values

        # 3) Evaluate model for each local sample, write local Y to files
        local_Y = np.zeros(local_param_values.shape[0])

        suffix = self._rep_suffix()
        y_local_file = f"{self._setting.output}_{method}{suffix}_Y_rank{self.rank}.csv"

        with open(y_local_file, "w") as f_y:
            f_y.write("global_idx,Y\n")
            for ii, i_global in enumerate(local_idx):
                pars = local_param_values[ii, :]
                self.update_parameters(pars)

                sim_values = self.simulation()

                # treat any NaN in simulation as fatal for this run
                if np.any( np.isnan(sim_values)):
                    msg = (
                        f"[{method}] Simulation produced NaN on rank {self.rank} "
                        f"for global index {int(i_global)}. "
                        "Aborting sensitivity analysis."
                    )
                    kklog_warn( msg)
                    if self.parallel:
                        MPI.COMM_WORLD.Abort( 1)
                    else:
                        sys.exit(1)

                # Compute scalar metric
                if output_metric == 'rmse':
                    val = self.objectivefunction( self._simulation, self._evaluation)
                if output_metric == 'rrmse':
                    val = self.objectivefunction( self._simulation, self._evaluation)
                elif output_metric == 'mean':
                    val = np.nanmean( sim_values)
                else:
                    raise ValueError(f"Unknown output_metric: {output_metric}")

                local_Y[ii] = val
                kklog_debug(
                    f"[run_sensitivity] rank={self.rank}, "
                    f"gi={int(i_global)}, Y={val:.15g}"
                )

                # write (global_index, value) to rank-specific file
                f_y.write(f"{int(i_global)},{local_Y[ii]:.15g}\n")

                if (ii + 1) % 10 == 0 or ii == len(local_idx) - 1:
                    kklog_info(
                        f"SALib (rank {self.rank}): "
                        f"finished {ii+1}/{len(local_idx)} local runs "
                        f"(global up to {i_global+1}/{n_runs})"
                    )

        if self.parallel:
            self.comm.Barrier()

        # 4) After all ranks are done: only rank 0 merges Y files and analyzes
        if self.parallel and self.rank != 0:
            return

        # Rank 0: merge rank Y files into one, sorted by global_idx
        merged_y_file = f"{self._setting.output}_{method}{suffix}_Y.csv"

        Y = np.empty(n_runs, dtype=float)
        Y[:] = np.nan

        rank_files = [
            f"{self._setting.output}_{method}{suffix}_Y_rank{r}.csv"
            for r in range(self.size)
        ]

        for rf in rank_files:
            if not os.path.exists(rf):
                kklog_warn(f"[{method}] Expected local Y file missing: {rf}")
                continue
            try:
                df_rf = pd.read_csv(rf)
            except pd.errors.EmptyDataError:
                kklog_warn(f"[{method}] Local Y file is empty and will be skipped: {rf}")
                continue
            except Exception as e:
                kklog_warn(f"[{method}] Error reading local Y file '{rf}': {repr(e)}")
                continue

            if df_rf.empty:
                kklog_warn(f"[{method}] Local Y file has no rows and will be skipped: {rf}")
                continue

            if not {"global_idx", "Y"}.issubset(df_rf.columns):
                kklog_warn(
                    f"[{method}] Local Y file missing required columns in '{rf}': "
                    f"found columns {list(df_rf.columns)}"
                )
                continue

            for _, row in df_rf.iterrows():
                gi = int(row["global_idx"])
                Y[gi] = float(row["Y"])

        missing = np.where(np.isnan(Y))[0]
        kklog_debug(
            f"[merge_Y] rank={self.rank}, n_runs={n_runs}, "
            f"missing_count={len(missing)}, "
            f"first_missing={missing[:20]}"
        )

        # Save merged Y to a single file, sorted by global_idx
        df_y = pd.DataFrame({
            "global_idx": np.arange(n_runs),
            "Y": Y
        })
        df_y_sorted = df_y.sort_values(by="global_idx")
        df_y_sorted.to_csv(merged_y_file, index=False)

        param_file = f"{self._setting.output}_{method}{suffix}_params.npy"
        if self.rank == 0:
            np.save(param_file, param_values)

        # Remove rank-specific files
        #for rf in rank_files:
        #    if os.path.exists(rf):
        #        try:
        #            os.remove(rf)
        #        except Exception as e:
        #            kklog_warn(f"[{method}] Could not remove local Y file '{rf}': {repr(e)}")

    @property
    def parallel( self) :
        return self._parallel

    def parameters( self) :
        return spotpy.parameter.generate( self.params)

    ## strip off units, e.g., "colX[kgm-2]" -> "colX"
    def canonicalize_headernames( self, _querydata) :
        data_columns = _querydata.columns
        unit_offs = lambda L, pos : L if pos == -1 else pos
        _querydata.columns = [ c[:unit_offs( len(c), c.find( '['))] for c in data_columns ]
        return _querydata

    def get_data( self, _target, _index=None):
        data_out = pd.DataFrame()
        calibrations = self._setting.calibrations

        for i, calib in enumerate(calibrations):
            target_cfg = calib[_target]
            expression = target_cfg['expression']
            variables = target_cfg['variables']

            # 1) For each variable, run provider if needed and read data
            eval_data = None
            for var in variables:
                ds = var['datasource']
                entity = var['entity']
                ds_name = ds.name

                if ds.has_provider:
                    ds.provider.execute()

                path = ds.path
                if _target == "simulation":
                    path = self._rank_specific_path(path)

                if not os.path.exists(path):
                    kklog_error(
                        f"File not found for calibration index {i}, "
                        f"target='{_target}': {path}"
                    )
                    sys.exit(255)

                data = pd.read_csv(
                    path, header=0,
                    na_values=["-99.99", "na", "nan"],
                    comment="#",
                    sep="\t",
                )
                data = self.canonicalize_headernames(data)

                if 'datetime' not in data.columns:
                    kklog_error(
                        f"'datetime' column missing in file:\n  {path}\n"
                        f"  columns: {list(data.columns)}"
                    )
                    sys.exit(255)

                # time subsetting
                if 'sampletime' in calib:
                    sampletime = calib['sampletime']
                    try:
                        t_from, t_to = sampletime.split("->")
                    except ValueError:
                        kklog_error(
                            f"Invalid sampletime format in calibration index {i}: "
                            f"'{sampletime}', expected 'YYYY-MM-DD->YYYY-MM-DD'"
                        )
                        sys.exit(255)
                    data = data.loc[(data['datetime'] >= t_from) & (data['datetime'] <= t_to), :]
                    data = data.set_index('datetime')
                    data.index = pd.to_datetime(data.index)
                else:
                    data = data.set_index('datetime')
                    data.index = pd.to_datetime(data.index)

                # optional filter
                if 'filter' in target_cfg:
                    for f in target_cfg['filter']:
                        for k, v in f.items():
                            data = data.loc[data[k].isin(v), :]

                if entity not in data.columns:
                    kklog_error(
                        f"Entity '{entity}' not in columns for calibration index {i}, "
                        f"target='{_target}'.\n  path: {path}\n  columns: {list(data.columns)}"
                    )
                    sys.exit(255)

                # collect this variable
                col = data[[entity]]
                col.columns = [entity]  # ensure column name consistent

                if eval_data is None:
                    eval_data = col
                else:
                    # align on time index, keep all variables
                    eval_data = eval_data.join(col, how="outer")

            # 2) Replace <entity>@<datasource_name> tokens in expression
            expr_eval = expression
            for var in variables:
                entity = var['entity']
                ds_name = var['datasource_name']
                token = f"{entity}{DSSEP}{ds_name}"
                if token in expr_eval:
                    expr_eval = expr_eval.replace(
                        token,
                        f'eval_data["{entity}"]'
                    )

            # 3) Evaluate expression
            try:
                result = eval(expr_eval).to_frame()
            except Exception as e:
                kklog_error(
                    f"Error evaluating expression '{expr_eval}': {repr(e)}\n"
                    f"{eval_data.head()}"
                )
                sys.exit(255)

            # 4) Rename to calibration id and append
            calib_id = calib['id']
            result.columns = [calib_id]
            data_out = pd.concat([data_out, result], axis=1)

        # aggregate across calibrations
        if _index is not None:
            collect_data = pd.DataFrame()
            for c in data_out.columns:
                column_data_out = data_out.loc[:, c].to_frame().dropna()
                column_data_index = _index.loc[:, c].to_frame().dropna()
                column_data = column_data_out.loc[
                    column_data_out.index.isin(column_data_index.index), :
                ]
                collect_data = pd.concat([collect_data, column_data], axis=1)
            return collect_data
        else:
            return data_out

    @property
    def dbname( self) :
        suffix = self._rep_suffix()
        return f"{self._setting.output}{suffix}"

    @property
    def method( self) :
        return self._setting.method

    @property
    def likelihood( self) :
        return self._setting.likelihood

    @property
    def repetitions( self) :
        return self._setting.repetitions

    def objectivefunction(self, simulation, evaluation):
        """
        Compute the objective function value.

        - If simulation and evaluation are DataFrames:
          use per-column logic (one value per calibration id) and return the mean.
        - If both are 1D NumPy arrays:
          compute a single objective value directly.
        - Otherwise: raise an error.
        """
        # --- DataFrame case: per-calibration logic ---
        if isinstance(simulation, pd.DataFrame) and isinstance(evaluation, pd.DataFrame):
            L = np.array([])

            for c in evaluation.columns:
                if c not in simulation.columns:
                    kklog_warn(f"Column '{c}' missing in simulation; skipping in objectivefunction.")
                    continue

                obs = evaluation[c].dropna().to_numpy()
                sim = simulation[c].dropna().to_numpy()

                n_obs = len(obs)
                n_sim = len(sim)

                if n_obs != n_sim:
                    raise ValueError(
                        f"Column '{c}': evaluation ({n_obs}) and simulation ({n_sim}) "
                        "have different lengths after removing NaNs."
                    )

                if self.objective_function == 'r2':
                    if n_obs < 3:
                        raise ValueError(
                            f"Column '{c}': r2 requires at least 3 values, got {n_obs}."
                        )
                    val = spotpy.objectivefunctions.rsquared( obs, sim)
                elif self.objective_function == 'rmse':
                    # negative rmse -> maximize
                    val = -spotpy.objectivefunctions.rmse( obs, sim)
                elif self.objective_function == 'rrmse':
                    # negative rrmse -> maximize
                    val = -spotpy.objectivefunctions.rrmse( obs, sim)
                elif self.objective_function == 'mean':
                    val = np.mean(sim)
                else:
                    raise ValueError(f"Unknown output metric: {self.objective_function}")

                L = np.append(L, val)

            if L.size == 0:
                raise ValueError("No overlapping calibration columns for objectivefunction.")

            self.likes.append(np.append(L, L.mean()))
            return L.mean()

        # --- NumPy array case: single series ---
        elif isinstance(simulation, np.ndarray) and isinstance(evaluation, np.ndarray):
            sim = simulation
            obs = evaluation

            if sim.ndim != 1 or obs.ndim != 1:
                raise ValueError(
                    "For NumPy inputs, simulation and evaluation must be 1D arrays "
                    f"(got sim.ndim={sim.ndim}, eval.ndim={obs.ndim})"
                )
            if len(sim) != len(obs):
                raise ValueError(
                    f"Length mismatch in objectivefunction: "
                    f"len(evaluation)={len(obs)}, len(simulation)={len(sim)}"
                )

            if self.objective_function == 'r2':
                val = spotpy.objectivefunctions.rsquared( obs, sim)
            elif self.objective_function == 'rmse':
                val = -spotpy.objectivefunctions.rmse( obs, sim)
            elif self.objective_function == 'rrmse':
                val = -spotpy.objectivefunctions.rrmse( obs, sim)
            elif self.objective_function == 'mean':
                val = np.mean(sim)
            else:
                raise ValueError(f"Unknown output metric: {self.objective_function}")

            self.likes.append(np.array([val, val]))
            return val

        else:
            raise TypeError(
                "objectivefunction expects either (DataFrame, DataFrame) or "
                "(ndarray, ndarray) as (simulation, evaluation), "
                f"got {type(simulation)} and {type(evaluation)}"
            )

    def run_simulation( self):
        """
        Run the external model(s) defined in self._setting.properties['model'].

        Returns
        -------
        rc : int
            Aggregate return code (0 if all commands succeeded, >0 otherwise).
        runtime : float
            Wall-clock time in seconds.
        """
        model_cfg = self._setting.properties.get("model", None)
        if model_cfg is None:
            kklog_warn("No 'model' configuration found in settings; nothing to run")
            return 0, 0.0

        program = os.path.expandvars(model_cfg["binary"])
        calls = model_cfg.get("calls", [])
        if not calls:
            kklog_warn("Model configuration has no 'calls'; nothing to run")
            return 0, 0.0

        # Build list of full commands
        model_calls = []
        for call in calls:
            call_expanded = os.path.expandvars(call)

            # handle rank-specific resources
            if self.parallel:
                rank = self.rank + 1
                call_expanded = call_expanded.replace("RANK", f"r{rank}")
            else:
                call_expanded = call_expanded.replace("RANK", "r1")

            cmd = f"{program} {call_expanded} > /dev/null 2>&1"
            model_calls.append(cmd)
            kklog_debug(f"Model call: {cmd}")

        t0 = time.time()
        return_codes = []

        # Execute each command sequentially
        for cmd in model_calls:
            try:
                proc = subprocess.Popen(cmd, shell=True)
            except FileNotFoundError:
                kklog_warn(f"Executable not found when running: {cmd}")
                return 1, 0.0
            except Exception as e:
                kklog_warn(f"Error starting process '{cmd}': {repr(e)}")
                return 1, 0.0

            rc = proc.wait()
            return_codes.append(rc)

            if rc != 0:
                kklog_warn(f"Model call failed with rc={rc}: {cmd}")

        t1 = time.time()
        runtime = round(t1 - t0, 2)

        max_rc = max(return_codes) if return_codes else 0
        if max_rc != 0:
            kklog_warn(
                f"One or more model calls failed, return codes: {return_codes}"
            )

        return max_rc, runtime

        # Aggregate return codes
        # if any command failed (rc != 0), we treat this as failure
        max_rc = max(return_codes) if return_codes else 0
        if max_rc != 0:
            kklog_warn(
                f"One or more model calls failed, return codes: {return_codes}"
            )

        return max_rc, runtime

    def update_parameters( self, _parameters=None):
        editor = self._setting.properties['model']['agent']
        L_input = os.path.expandvars(editor['in'])
        base_out = os.path.expandvars(editor['out'])

        rank_suffix = f"r{self.rank + 1}" if self.parallel else "r1"
        if "RANK" in base_out:
            L_output = base_out.replace("RANK", rank_suffix)
        else:
            L_output = f"{base_out}_{rank_suffix}"

        # read template Lresources
        with open(f"{L_input}/Lresources", "r") as f:
            subject = f.read()

        if _parameters is not None:
            p_index = 0
            for key, v in self._setting.parameters.items():
                pname = v["name"]
                target = v.get("target", "")
                species = v.get("species", None)

                if target.lower() in ("siteparameter", "siteparameters", "site"):
                    left_pattern = rf"site\.parameter\.{re.escape(pname)}\.value"
                elif target.lower() in ("speciesparameter", "speciesparameters", "species"):
                    if species is None:
                        kklog_warn(
                            f'Parameter "{key}" target is "species" but no "species" id is given; '
                            f'skipping.'
                        )
                        p_index += 1
                        continue
                    left_pattern = rf"species\.{re.escape(species)}\.parameter\.{re.escape(pname)}\.value"
                else:
                    kklog_warn(
                        f'Parameter "{key}" has unknown target "{target}"; skipping.'
                    )
                    p_index += 1
                    continue

                pattern = re.compile(
                    rf'^({left_pattern})\s*=\s*".*?"\s*$',
                    re.MULTILINE
                )
                match = pattern.search(subject)
                if match is None:
                    kklog_warn(
                        f'Parameter "{key}" (target="{target}") not found in Lresources; '
                        f'no replacement performed.'
                    )
                else:
                    left_side = match.group(1)
                    val = _parameters[p_index]
                    new_line = f'{left_side} = "{val:.15g}"'
                    subject = pattern.sub(new_line, subject, count=1)

                p_index += 1

        # write updated Lresources (rank-specific)
        if not os.path.exists(L_output):
            os.makedirs(L_output)

        # copy udunits2 directory once
        src_udunits = os.path.join(L_input, "udunits2")
        dst_udunits = os.path.join(L_output, "udunits2")
        if os.path.exists(src_udunits) and not os.path.exists(dst_udunits):
            shutil.copytree(src_udunits, dst_udunits)

        with open(f"{L_output}/Lresources", "w") as f:
            f.write(subject)

    def simulation( self, _parameters=None):
        # 1) Update parameters file if new parameters are given
        if _parameters is not None:
            self.update_parameters(_parameters)

        # 2) Run the model
        rc, runtime = self.run_simulation()
        kklog_debug(f"Simulation duration {runtime} s")

        # 3) If model call clearly failed (non-zero return code), avoid reading data
        if rc > 0:
            kklog_warn(
                f"Model call returned non-zero exit code (rc={rc}) "
                "– filling simulation with NaNs"
            )
            # Build a NaN DataFrame with the same index and columns as evaluation
            if not isinstance(self._evaluation, pd.DataFrame):
                raise RuntimeError(
                    "simulation(): self._evaluation is not a DataFrame; "
                    "cannot construct NaN simulation of matching shape."
                )

            self._simulation = pd.DataFrame(
                np.nan,
                index=self._evaluation.index,
                columns=self._evaluation.columns,
            )
            # Return stacked NaNs (SpotPy expects a 1D array)
            return self._simulation.stack( future_stack=True).dropna().to_numpy()

        # 4) Try to read simulation output (wide DataFrame)
        try:
            self._simulation = self.get_data("simulation", self._evaluation)  # wide: columns = calib ids
        except SystemExit:
            raise
        except Exception as e:
            kklog_warn(
                f"Unexpected error while loading simulation data: {repr(e)} "
                "– filling simulation with NaNs"
            )
            if not isinstance(self._evaluation, pd.DataFrame):
                raise RuntimeError(
                    "simulation(): self._evaluation is not a DataFrame; "
                    "cannot construct NaN simulation of matching shape."
                )
            self._simulation = pd.DataFrame(
                np.nan,
                index=self._evaluation.index,
                columns=self._evaluation.columns,
            )

        # 5) Return 1D numpy array for SpotPy / SALib:
        transposed = self._simulation.T
        return transposed.stack( future_stack=True).dropna().to_numpy()

    def evaluation( self):
        transposed = self._evaluation.T
        return transposed.stack( future_stack=True).dropna().to_numpy()

    def evaluation_df( self):
        # Transpose the DataFrame to order values column-wise
        transposed = self._evaluation.T
        stacked = transposed.stack(future_stack=True).dropna()
        df_stacked = stacked.to_frame(name='value').reset_index()
        df_stacked.columns = ['calibration_id', 'time', 'value']
        return df_stacked

    def last_simulation_df( self):
        # Transpose the DataFrame to order values column-wise
        transposed = self._simulation.T
        stacked = transposed.stack(future_stack=True).dropna()
        df_stacked = stacked.to_frame(name='value').reset_index()
        df_stacked.columns = ['calibration_id', 'time', 'value']
        return df_stacked

    #write spoty output more userfriendly
    def finalize( self, _sampler) :
        pass
        #results = _sampler.getdata()
        #try:
        #    spotpy.analyser.plot_fast_sensitivity( results, number_of_sensitiv_pars=3)
        #except:
        #    pass

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parallel = comm.Get_size() > 1

    kkplot_env = kkexpand('${HOME}') + '/.kkplot/kkplot.env'
    if exists(kkplot_env):
        load_dotenv(kkplot_env)
    kkplot_env = kkexpand('${HOME}') + '/.ldndc/kkplot.env'
    if exists(kkplot_env):
        load_dotenv(kkplot_env)

    config = utils.configuration()
    project = kkopt_project( config, _parallel=parallel)

    if not config.nosim():
        setup = spot_setup( config, project)

        if setup.method in ['mcmc', 'fast', 'lhs']:
            lspotpy_functions = {
                'lhs': spotpy.algorithms.lhs,
                'fast': spotpy.algorithms.fast,
                'mcmc': spotpy.algorithms.mcmc,
            }
            if project.parallel:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=setup.dbname,
                    dbformat=project.setting.outputformat,
                    parallel='mpi',
                )
            else:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=setup.dbname,
                    dbformat=project.setting.outputformat,
                )
            sampler.sample(setup.repetitions)
            setup.finalize(sampler)

        elif setup.method in ['sobol', 'morris']:
            setup.run_sensitivity( method=setup.method, N=setup.repetitions)
            if project.parallel:
                kklog_info( f"Rank {rank + 1} terminated successfully!")
        else:
            raise ValueError(f"Unknown method: {setup.method}")

    # Only rank 0 runs postprocessing
    if rank == 0:
        postprocess( project)


if __name__ == '__main__':
    main()
