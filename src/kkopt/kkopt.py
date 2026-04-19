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
from kkopt.kkopt_postprocess import postprocess

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
        self.simulation_counter = 0

        #prepare evaluation data
        temp = self.get_data( 'simulation')
        self._evaluation = self.get_data( 'evaluation', temp)
        self._simulation = self.get_data( 'simulation', self._evaluation)

        df_tmp = pd.concat([self._evaluation['all'], self._simulation['all']], axis=1, keys=['evaluation', 'simulation'])

        output_path = f"{self._setting.output}_base.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_tmp.to_csv( output_path)

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
            return base_path

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
            f"[get_data] No rank-specific file found, using base file instead:\n"
            f"  rank={rank}\n  base: {base_path}"
        )
        return base_path

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
                print(
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
                    val = self.objectivefunction(self._simulation, self._evaluation)
                elif output_metric == 'mean':
                    val = np.nanmean(sim_values)
                else:
                    raise ValueError(f"Unknown output_metric: {output_metric}")

                local_Y[ii] = val

                # write (global_index, value) to rank-specific file
                f_y.write(f"{int(i_global)},{local_Y[ii]:.15g}\n")

                if (ii + 1) % 10 == 0 or ii == len(local_idx) - 1:
                    print(
                        f"SALib (rank {self.rank}): "
                        f"finished {ii+1}/{len(local_idx)} local runs "
                        f"(global up to {i_global+1}/{n_runs})"
                    )

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
        for rf in rank_files:
            if os.path.exists(rf):
                try:
                    os.remove(rf)
                except Exception as e:
                    kklog_warn(f"[{method}] Could not remove local Y file '{rf}': {repr(e)}")


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

    def get_data( self, _target, _index=None) :

        data_out = pd.DataFrame()

        calibrations = self._setting.calibrations

        for i, calib in enumerate( calibrations):

            # ------------------------------------------------------------------
            # 1) Run provider if present
            # ------------------------------------------------------------------
            if calib[_target]['datasource'].has_provider:
                calib[_target]['datasource'].provider.execute()
            datasource_name = calib[_target]['datasource'].name

            entity = calib[_target]['entity']
            path = calib[_target]['datasource'].path

            # ------------------------------------------------------------------
            # 2) Resolve path for MPI / rank
            # ------------------------------------------------------------------
            path = self._setting.calibrations[i][_target]['datasource'].path
            if self.parallel and _target == "simulation":
                path = self._rank_specific_path(path)

            if not os.path.exists(path):
                print(
                    f"[get_data] File not found for calibration index {i}, "
                    f"target='{_target}': {path}"
                )
                sys.exit(255)

            # ------------------------------------------------------------------
            # 3) Read raw data
            # ------------------------------------------------------------------
            try:
                data = pd.read_csv(
                    path,
                    header=0,
                    na_values=["-99.99", "na", "nan"],
                    comment="#",
                    sep="\t",
                )
            except Exception as e:
                print(
                    f"[get_data] Error reading CSV for calibration index {i}, "
                    f"target='{_target}'\n  path: {path}\n  error: {repr(e)}"
                )
                sys.exit(255)

            data = self.canonicalize_headernames(data)

            if "datetime" not in data.columns:
                print(
                    f"[get_data] 'datetime' column missing in file:\n  {path}\n"
                    f"  columns: {list(data.columns)}"
                )
                sys.exit(255)

            # ------------------------------------------------------------------
            # 4) Time subsetting (sampletime)
            # ------------------------------------------------------------------
            if "sampletime" in calib:
                sampletime = calib["sampletime"]
                try:
                    t_from, t_to = sampletime.split("->")
                except ValueError:
                    print(
                        f"[get_data] Invalid sampletime format in calibration index {i}: "
                        f"'{sampletime}', expected 'YYYY-MM-DD->YYYY-MM-DD'"
                    )
                    sys.exit(255)
                eval_data = data.loc[(data['datetime'] >= t_from) & (data['datetime'] <= t_to),]
                eval_data = eval_data.set_index('datetime')
                eval_data.index = pd.to_datetime(eval_data.index)
            else:
                eval_data = data

            # ------------------------------------------------------------------
            # 5) Optional filtering by columns
            # ------------------------------------------------------------------
            if 'filter' in calib[_target]:
                for f in calib[_target]['filter']:
                    for k,v in f.items():
                        eval_data = eval_data.loc[eval_data[k].isin(v),]

            # ------------------------------------------------------------------
            # 6) Select the entity column
            # ------------------------------------------------------------------
            if entity not in eval_data.columns:
                print(
                    f"[get_data] Entity '{entity}' not in columns for calibration index {i}, "
                    f"target='{_target}'.\n  path: {path}\n  columns: {list(eval_data.columns)}"
                )
                sys.exit(255)
            eval_data = eval_data[[entity]]

            # ------------------------------------------------------------------
            # 7) Apply expression
            # ------------------------------------------------------------------
            expression = calib[_target]['expression']
            expression = expression.replace( entity+DSSEP+datasource_name, 'eval_data["%s"]' %entity)
            try:
                eval_data = eval(expression).to_frame()
            except TypeError:
                print( f"TypeError: {expression}\n{eval_data.head()}")
                sys.exit( 255)

            # ------------------------------------------------------------------
            # 8) Rename column to calibration id and append
            # ------------------------------------------------------------------
            calib_id = calib["id"]
            eval_data.rename(columns={entity: calib_id}, inplace=True)
            data_out = pd.concat([data_out, eval_data])

        if _index is not None:
            collect_data = pd.DataFrame()
            for c in data_out.columns:
                column_data_out = data_out.loc[:,c].to_frame().dropna()
                column_data_index = _index.loc[:,c].to_frame().dropna()
                column_data = column_data_out.loc[column_data_out.index.isin(column_data_index.index),:]
                collect_data = pd.concat([collect_data, column_data])
            collect_data['all'] = collect_data.sum(axis=1)
            return collect_data

        data_out['all'] = data_out.sum(axis=1)

        return data_out

    @property
    def dbname( self) :
        return self._setting.output

    @property
    def method( self) :
        return self._setting.method

    @property
    def repetitions( self) :
        return self._setting.repetitions

    # This function is needed for spotpy to compare simulation and validation data
    # Keep in mind, that you reduce your simulation data to the values that can be compared with observation data
    # This can be done in the def simulation (than only those simulations are saved), 
    # or in the def objectivefunctions (than all simulations are saved)
    def objectivefunction( self, simulation, evaluation) :

        L = np.array([])
        for c in self._evaluation.columns:
            if c == 'all':
                continue
            if self.objective_function == 'r2' :
                L = np.append(L, spotpy.objectivefunctions.rsquared( self._evaluation[c].dropna().squeeze().values,
                                                                     self._simulation[c].dropna().squeeze().values))
            elif self.objective_function == 'rmse' :
                L = np.append(L, -spotpy.objectivefunctions.rmse( self._evaluation[c].dropna().squeeze().values,
                                                                  self._simulation[c].dropna().squeeze().values))
            elif self.objective_function == 'mean' :
                L = np.append(L, np.mean( self._simulation[c].dropna().squeeze().values))
            else:
                raise ValueError(f"Unknown output metric: {self.objective_function}")

        self.likes.append( np.append( L, L.mean()))
        return L.mean()

    def run_simulation(self):
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

    def update_parameters(self, _parameters=None):
        editor = self._setting.properties['model']['agent']
        L_input = os.path.expandvars(editor['in'])
        L_output = os.path.expandvars(editor['out'])

        # Handle rank-specific output path
        if self.parallel:
            rank = self.rank + 1
            L_output = L_output.replace("RANK", f"r{rank}")
        else:
            L_output = L_output.replace("RANK", "r1")

        with open(f"{L_input}/Lresources", "r") as f:
            subject = f.read()

        if _parameters is not None:
            p_index = 0
            for key, v in self._setting.parameters.items():
                pname = v["name"]  # the name used inside Lresources

                pattern = re.compile(
                    rf'^(.*\.{re.escape(pname)}\..*?)\s*=\s*".*?"\s*$',
                    re.MULTILINE
                )
                match = pattern.search(subject)

                if match is None:
                    kklog_warn(
                        f'Parameter "{pname}" not found in Lresources; '
                        f'no replacement performed.'
                    )
                else:
                    left_side = match.group(1)
                    val = _parameters[p_index]
                    # high precision, scientific or fixed as needed
                    new_line = f'{left_side} = "{val:.15g}"'
                    subject = pattern.sub(new_line, subject, count=1)

                p_index += 1


        if not os.path.exists(L_output):
            os.makedirs(L_output)

        # copy udunits2 directory once
        src_udunits = os.path.join(L_input, "udunits2")
        dst_udunits = os.path.join(L_output, "udunits2")
        if os.path.exists(src_udunits) and not os.path.exists(dst_udunits):
            shutil.copytree(src_udunits, dst_udunits)

        with open(f"{L_output}/Lresources", "w") as f:
            f.write(subject)

    def simulation( self, _parameters=None) :

        # 1) Update parameters file if new parameters are given
        if _parameters is not None:
            self.update_parameters(_parameters)

        # 2) Run the model
        run_id = self.simulation_counter + 1

        if self.parallel:
            kklog_debug(f"Rank {self.rank + 1}")
        else:
            kklog_debug(f"Model run {run_id}")

        rc, runtime = self.run_simulation()
        kklog_info(f"Rank {self.rank + 1}  rc={rc}, runtime={runtime}")
        self.simulation_counter = run_id
        kklog_debug(f"Simulation duration {runtime} s")

        # 3) If model call clearly failed (non-zero return code), avoid reading data
        if rc > 0:
            kklog_warn(
                f"Model call returned non-zero exit code (rc={rc}) "
                "– filling simulation with NaNs"
            )
            # Build a NaN series with the same index as evaluation to keep shapes consistent
            sim_nan = pd.Series(
                np.nan, index=self._evaluation.index, name="all"
            )
            self._simulation = pd.DataFrame(sim_nan)
            return self._simulation["all"].to_numpy()

        # 4) Try to read simulation output
        try:
            sim = self.get_data("simulation", self._evaluation)
        except SystemExit:
            # get_data already printed an error; propagate
            raise
        except Exception as e:
            kklog_warn(
                f"Unexpected error while loading simulation data: {repr(e)} "
                "– filling simulation with NaNs"
            )
            sim = pd.DataFrame(
                np.nan,
                index=self._evaluation.index,
                columns=["all"],
            )

        # 5) Sanity check: non-empty, correct column
        if not isinstance(sim, pd.DataFrame) or "all" not in sim.columns:
            kklog_warn(
                "Loaded simulation data has unexpected structure; "
                "expected DataFrame with column 'all'. "
                "Filling with NaNs."
            )
            sim = pd.DataFrame(
                np.nan,
                index=self._evaluation.index,
                columns=["all"],
            )

        self._simulation = sim

        # 6) Return 1D numpy array for SpotPy / SALib
        return self._simulation["all"].to_numpy()

    def evaluation( self):
        return self._evaluation['all'].squeeze().values

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
    project = kkopt_project(config, _parallel=parallel)

    if not config.nosim():
        setup = spot_setup(config, project)

        if setup.method in ['mcmc', 'fast', 'lhs']:
            lspotpy_functions = {
                'lhs': spotpy.algorithms.lhs,
                'fast': spotpy.algorithms.fast,
                'mcmc': spotpy.algorithms.mcmc,
            }
            if project.parallel:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=project.setting.output,
                    dbformat=project.setting.outputformat,
                    parallel='mpi',
                )
            else:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=project.setting.output,
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
