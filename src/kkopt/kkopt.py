#!/bin/bash

import sys
import os
from os.path import exists
from dotenv import load_dotenv
import time
import re
import pandas as pd
import numpy as np
import numexpr as numexpr
import spotpy
from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris as morris_analyze
import subprocess
try:
    from mpi4py import MPI
except ImportError:
    raise Exception("MPI python module mpi4py not available. Exit!")

from kkplot.kkutils.expand import *
from kkplot.kkutils.log import *
from kkplot.kksources import kkplot_sourcefactory as kkplot_sourcefactory
from kkplot.kkplot_dviplot import *
from kkplot.kkplot_figure import DSSEP

import kkopt.kkutils as utils
from kkopt.kkopt_project import kkopt_project 
from kkopt.kkopt_postprocess import postprocess


class lspotpy_object_factories( object) :
    def  __init__( self, _kinds) :
        self._kinds = _kinds
        self.factories = dict()
    @property
    def  kinds( self) :
        return  self._kinds

    def  register( self, _kind, _obj) :
        self.factories[_kind] = _obj
    def  create( self, _kind, **_kwargs) :
        if _kind in self.factories :
            return self.factories[_kind].create( _kwargs)
        return None
LSPOTPY_INTERPOLATION_FACTORIES = lspotpy_object_factories( 'interpolators')

class lspotpy_interpolation( object) :
    def  __init__( self) :
        pass
    def interpolate( self, _num) :
        raise NotImplementedError( 'missing implementation')

class lspotpy_interpolation_factory( object) :
    def  __init__( self, _kind) :
        self.kind = _kind
        LSPOTPY_INTERPOLATION_FACTORIES
        if not self.kind in LSPOTPY_INTERPOLATION_FACTORIES :
            LSPOTPY_INTERPOLATION_FACTORIES[self.kind] = self

class lspotpy_interpolation_linear( lspotpy_interpolation) :
    def  __init__( self, minvalue=numpy.nan, maxvalue=numpy.nan) :
        super( lspotpy_interpolation_linear, self)
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def  interpolate( self, _num) :
        return list( numpy.linspace( self.minvalue, self.maxvalue, _num))
 
class lspotpy_interpolation_factory_linear( lspotpy_interpolation_factory) :
    def  __init__( self, _kind) :
        super( lspotpy_interpolation_factory_linear, self, _kind)
    def  create( self, **_kwargs) :
        return  lspotpy_interpolation_linear( _kwargs)
#__lspotpy_interpolation_factory_linear = lspotpy_interpolation_factory_linear( 'linear')

## Distribution
class lspotpy_distribution( object) :
    def  __init__( self, _kind, **_kwargs) :
        self.kind = _kind

        self.minvalues = None
        self.maxvalues = None
        self.values = None

        if 'minvalues' in _kwargs :
            self.minvalues = _kwargs['minvalues']
            if not isinstance( self.minvalues, list) :
                self.minvalues = [ self.minvalues]
        if 'maxvalues' in _kwargs :
            self.maxvalues = _kwargs['maxvalues']
            if not isinstance( self.maxvalues, list) :
                self.maxvalues = [ self.maxvalues]
        if 'values' in _kwargs :
            if self.minvalues is not None or self.maxvalues is not None :
                raise RuntimeError( 'arguments "minvalues", "maxvalues" or "values" are mutually exclusive')
            self.values = _kwargs['values']

        if 'interpolation' in _kwargs :
            self.interpolation = lspotpy_interpolations.create( _kwargs['interpolation'])

        if self.kind == 'uniform' :
            self.distribution = numpy.random.uniform

    def  sample_continuum( self) :
        samples = list()
        for mi, ma in zip( self.minvalues, self.maxvalues) :
            samples.append( self.distribution( mi, ma))

    def  sample( self, _num) :
        samples = list()
        for n in xrange( _num) :
            samples.append( self.sample_continuum())

def  lspotpy_makedistribution( _kind, **_kwargs) :
    if _kind in LSPOTPY_INTERPOLATION_FACTORIES :
        distribution = LSPOTPY_INTERPOLATION_FACTORIES.create( _kind, _kwargs)

class lspotpy_parameter( object) :
    def  __init__( self, _name) :
        self.name = _name

        self.values = list()
        self.minvalue = numpy.nan
        self.maxvalue = numpy.nan
        self.initialvalue = numpy.nan
        self.n_values = 0

        self.distribution = lspotpy_makedistribution( _distribution, _properties)

    def  sample( self) :
        return  self.distribution.sample( self.n_samples)


class spot_setup(object):
    def __init__(self, _config, _project):

        self._configuration = _config
        ##get all settings 
        self._setting = _project.setting
        self._parallel = _project.parallel

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
        #df_tmp.to_csv( f"{self._setting.output}_base.csv")

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

    def run_sensitivity( self, method='sobol', N=1000, output_metric='rmse'):
        """
        Run global sensitivity analysis using SALib.

        Parameters
        ----------
        method : str
            'sobol' or 'morris'.
        N : int
            Base sample size for SALib.
        output_metric : str
            'rmse' or 'mean' – how to reduce the time series to one value per run.
        """
        problem = self.build_salib_problem()
        n_pars = problem['num_vars']

        # 1) Generate samples
        if method == 'sobol':
            param_values = saltelli.sample( problem, N, calc_second_order=True)
        elif method == 'morris':
            # k is number of trajectories, p is grid levels
            k = N
            param_values = morris_sample.sample( problem, N=k, num_levels=4, optimal_trajectories=None)
        else:
            raise ValueError(f"Unknown SALib method: {method}")

        n_runs = param_values.shape[0]
        print(f"SALib: generated {n_runs} samples for method={method}")

        # 2) Evaluate model for each sample
        Y = np.zeros( n_runs)

        for i in range( n_runs):
            pars = param_values[i, :]
            # Update Lresources with current parameters
            self.update_parameters( pars)
            # Run model and read simulation
            self.simulation()

            Y[i] = self.objectivefunction( self._simulation, self._evaluation)

            if (i + 1) % 10 == 0 or i == n_runs - 1:
                print(f"SALib: finished {i+1}/{n_runs} runs")

        # 3) Remove NaNs if any runs failed
        valid_idx = ~np.isnan(Y)
        if not np.all(valid_idx):
            print(f"SALib: {np.sum(~valid_idx)} runs had NaN output, excluding them")
            param_values = param_values[valid_idx, :]
            Y = Y[valid_idx]

        # 4) Analyze
        if method == 'sobol':
            Si = sobol.analyze(
                problem, Y,
                print_to_console=True
            )
            # Save Sobol indices
            out_base = f"{self._setting.output}_sobol"
            np.savetxt(out_base + "_S1.csv",
                       np.vstack([problem['names'], Si['S1']]).T,
                       delimiter=",", fmt="%s")
            np.savetxt(out_base + "_ST.csv",
                       np.vstack([problem['names'], Si['ST']]).T,
                       delimiter=",", fmt="%s")
            if 'S2' in Si:
                # pairwise second-order
                # flatten into table (i, j, S2_ij)
                names = problem['names']
                S2_list = []
                for i in range(n_pars):
                    for j in range(i + 1, n_pars):
                        S2_list.append([names[i], names[j], Si['S2'][i, j]])
                S2_arr = np.array(S2_list, dtype=object)
                np.savetxt(out_base + "_S2.csv",
                           S2_arr,
                           delimiter=",", fmt="%s")

        elif method == 'morris':
            Si = morris_analyze.analyze(
                problem,
                param_values,
                Y,
                print_to_console=True
            )
            out_base = f"{self._setting.output}_morris"
            # mu*, sigma etc.
            arr = np.vstack([
                problem['names'],
                Si['mu_star'],
                Si['sigma'],
                Si['mu']
            ]).T
            header = "name,mu_star,sigma,mu"
            np.savetxt(out_base + "_indices.csv",
                       arr,
                       delimiter=",",
                       fmt="%s",
                       header=header,
                       comments="")
        else:
            raise ValueError(f"Unknown SALib method: {method}")

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
        for i in range(len(self._setting.calibrations)):
            if self._setting.calibrations[i][_target]['datasource'].has_provider:
                self._setting.calibrations[i][_target]['datasource'].provider.execute()
            datasource_name = self._setting.calibrations[i][_target]['datasource'].name

            entity = self._setting.calibrations[i][_target]['entity']
            path = self._setting.calibrations[i][_target]['datasource'].path
            if self.parallel:
                path = path + str(MPI.COMM_WORLD.Get_rank()+1)
                path = path.replace( "RANK", "r"+str(MPI.COMM_WORLD.Get_rank()+1))
            else:
                path = path.replace( "RANK", "r1")
            data = pd.read_csv( path, header=0, na_values=['-99.99','na','nan'], comment='#', sep="\t")
            data = self.canonicalize_headernames( data)

            if 'sampletime' in self._setting.calibrations[i]:
                sampletime = self._setting.calibrations[i]['sampletime']
                t_from, t_to = sampletime.split( '->')
                eval_data = data.loc[(data['datetime'] >= t_from) & (data['datetime'] <= t_to),]
                eval_data = eval_data.set_index('datetime')
                eval_data.index = pd.to_datetime(eval_data.index)
            else:
                eval_data = data

            if 'filter' in self._setting.calibrations[i][_target]:
                for f in self._setting.calibrations[i][_target]['filter']:

                    for k,v in f.items():

                        eval_data = eval_data.loc[eval_data[k].isin(v),]
            eval_data = eval_data[[entity]]


            expression = self._setting.calibrations[i][_target]['expression']
            expression = expression.replace( entity+DSSEP+datasource_name, 'eval_data["%s"]' %entity)
            try:
                eval_data = eval(expression).to_frame()
            except TypeError:
                print( f"TypeError: {expression}\n{eval_data.head()}")
                sys.exit( 255)
            eval_data.rename(columns={entity: self._setting.calibrations[i]['id']}, inplace=True)
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

    def run_simulation( self, _parallel = False) :

        models = [self._setting.properties['model']]

        for model in models:
            program = os.path.expandvars( model['binary'])
            model_calls = []
            for call in model['calls']:

                #argument = argument + os.path.expandvars( arg) + ' '
                if False: #self.parallel:
                    model_calls.append( program+" "+argument + str(MPI.COMM_WORLD.Get_rank()+1) + " > /dev/null 2>&1")
                    kklog_debug( f'Rank {str(MPI.COMM_WORLD.Get_rank())}: %s' %str(model_calls[-1]))
                else:
                    model_calls.append( program+" "+os.path.expandvars( call) + " > /dev/null 2>&1")
        t0 = time.time()

        # List to store subprocess objects
        processes = []

        # Start each command as a subprocess
        for m in model_calls:
            # Use shell=True to run the command through the shell
            process = subprocess.Popen( m, shell=True)
            processes.append(process)

        # Wait for all processes to complete
        rcs = np.array([])
        for process in processes:
            rcs = np.append( rcs, process.wait())

        t1 = time.time()

        return (rcs.sum(), round( (t1-t0),2))

    def update_parameters( self, _parameters=None):

        editor = self._setting.properties['model']['agent']
        provider = editor['provider']
        L_input = os.path.expandvars( editor['in'])
        L_output = os.path.expandvars( editor['out'])

        # open the source file and read it
        subject = ''
        #with open( kkexpand('${HOME}')+'/.ldndc/Lresources', 'r') as f:
        with open( f"{L_input}/Lresources", 'r') as f:
            subject = f.read()

        if _parameters is not None:
            p_index = 0

            for k,v in self._setting.parameters.items() :
                search = re.search(r'.*\.%s\..*' % v['name'], subject)
                #todo: add log warning
                if search != None:
                    pattern = re.compile(r'.*\.%s\..*' % v['name'])
                    subject = pattern.sub('%s = "%f"' %(search.group(0).split('=')[0], _parameters[p_index]), subject)
                p_index += 1

        # write the file
        import shutil
        if self.parallel:
            L_output = f'{L_output}_{str(MPI.COMM_WORLD.Get_rank()+1)}'
        #else:
        #    L_output = os.path.expanduser(kkexpand('${HOME}')+f'/.ldndc/{L_output}')
        if not os.path.exists( L_output):
            os.makedirs( L_output)
        if not os.path.exists( L_output+"/udunits2"):
            shutil.copytree( L_input+"/udunits2", L_output+"/udunits2")
        with open( f"{L_output}/Lresources", 'w') as f:
            f.write( subject)

    def simulation( self, _parameters=None) :

        if _parameters is not None:
            self.update_parameters( _parameters)

        kklog_debug('Model run %s' %str(self.simulation_counter+1))
        if self.parallel:
            kklog_info('Rank %s' %str( MPI.COMM_WORLD.Get_rank()+2))
        (rc,time) = self.run_simulation()
        self.simulation_counter += 1
        kklog_debug('Simulation duration %s s' %str(time))

        self._simulation = self.get_data( 'simulation', self._evaluation)
        if None in self._simulation :
            kklog_warn("loading of simulation data failed")
            self._simulation = pd.DataFrame( np.nan, index=range(self._simulation.shape[0]), columns=self._simulation.columns)
        elif rc > 0:
            kklog_warn("model call not successful")
            self._simulation = pd.DataFrame( np.nan, index=range(self._simulation.shape[0]), columns=self._simulation.columns)
        return self._simulation['all'].squeeze().values

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
    if comm.Get_size() > 1:
        parallel = True
    else:
        parallel = False
    kkplot_env = kkexpand( '${HOME}')+'/.kkplot/kkplot.env'
    if ( exists( kkplot_env)) :
        load_dotenv( kkplot_env)
    kkplot_env = kkexpand( '${HOME}')+'/.ldndc/kkplot.env'
    if ( exists( kkplot_env)) :
        load_dotenv( kkplot_env)

    ### Configuration object
    config = utils.configuration()

    ### Calibration project
    project = kkopt_project( config, _parallel = parallel)

    if not config.nosim():
        setup = spot_setup( config, project)

        # calibration (spotpy)
        if setup.method in ['mcmc', 'fast', 'lhs']:  # existing
            lspotpy_functions = dict({
                'lhs': spotpy.algorithms.lhs,
                'fast': spotpy.algorithms.fast,
                'mcmc': spotpy.algorithms.mcmc
            })
            if project.parallel:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=project.setting.output,
                    dbformat=project.setting.outputformat,
                    parallel='mpi'
                )
            else:
                sampler = lspotpy_functions[setup.method](
                    setup,
                    dbname=project.setting.output,
                    dbformat=project.setting.outputformat
                )
            sampler.sample( setup.repetitions)
            setup.finalize( sampler)

        # sensitivity (SALib)
        elif setup.method in ['sobol', 'morris']:
            setup.run_sensitivity( method=setup.method, N=setup.repetitions)
        else:
            raise ValueError(f"Unknown method: {setup.method}")

    postprocess(project)

if __name__ == '__main__' :

    main()
