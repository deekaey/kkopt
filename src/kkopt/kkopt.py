#!/bin/bash

import string
import sys
import os
from os.path import exists
from dotenv import load_dotenv
import time
import re
import pandas as pd
import numpy as np
import numexpr as numexpr
import math
import scipy
import spotpy
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
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
                self.params.append( spotpy.parameter.Uniform( k,
                                                          v['minvalue'],
                                                          v['maxvalue'],
                                                          v['initialvalue'],
                                                          v['step']))

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

        self.likes.append( np.append( L, L.mean()))
        return L.mean()

    def run_simulation( self, _parallel = False) :


        if 'submodels' in self._setting.properties['model']:
            models = self._setting.properties['model']['submodels']
        else:
            models = [self._setting.properties['model']]

        model_calls = []
        for submodel in models:
            program = os.path.expandvars( self._setting.properties['model']['binary'])
            argument = ''
            for arg in self._setting.properties['model']['arguments'] :
                argument = argument + os.path.expandvars( arg) + ' '
            if self.parallel:
                model_calls.append( program+" "+argument + str(MPI.COMM_WORLD.Get_rank()+1) + " > /dev/null 2>&1")
            else:
                model_calls.append( program+" "+argument + " > /dev/null 2>&1")
            kklog_debug( f'Rank {str(MPI.COMM_WORLD.Get_rank())}: %s' %str(model_calls[-1]))
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
        # open the source file and read it
        subject = ''
        with open( kkexpand('${HOME}')+'/.ldndc/Lresources', 'r') as f:
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
            Lresources_path = os.path.expanduser(kkexpand('${HOME}')+'/.ldndc/Lresources_tmp_'+str(MPI.COMM_WORLD.Get_rank()+1))
        else:
            Lresources_path = os.path.expanduser(kkexpand('${HOME}')+'/.ldndc/Lresources_tmp_1')
        if not os.path.exists( Lresources_path):
            os.makedirs( Lresources_path)
        if not os.path.exists( Lresources_path+"/udunits2"):
            shutil.copytree( kkexpand('${HOME}')+'/.ldndc/udunits2', Lresources_path+'/udunits2')
        with open(os.path.join( Lresources_path, 'Lresources'), 'w') as f:
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

def postprocess(project):

    base = pd.read_csv(f"{project.setting.output}_base.csv")
    base = base.set_index( pd.to_datetime(base.datetime))

    percentile_threshold = 0.2
    delimiter = ','
    observed_values = None
    observed_values = base['evaluation']
    like_type = "RMSE"

    # === DATEN EINLESEN ===
    df = pd.read_csv( project.output_file, delimiter=delimiter)

    # === SPALTEN IDENTIFIZIEREN ===
    like_col = 'like1'
    param_cols = [col for col in df.columns if col.startswith('par')]
    sim_cols = [col for col in df.columns if col.startswith('simulation_')]


    if like_type == "R2":
        df["R2"] = df[sim_cols].apply(lambda row: spotpy.objectivefunctions.rsquared(row.values, observed_values), axis=1)
        df_sorted = df.sort_values(by=like_col, ascending=False)
        top_n = int(len(df_sorted) * percentile_threshold)
        df_top = df_sorted.head(top_n)
    else:
        df["RMSE"] = df[sim_cols].apply(lambda row: spotpy.objectivefunctions.rmse(row.values, observed_values), axis=1)
        df_sorted = df.sort_values(by="RMSE", ascending=True)  # kleiner RMSE = besser
        top_n = int(len(df_sorted) * percentile_threshold)
        df_top = df_sorted.head(top_n)

    os.makedirs(project.output_dir, exist_ok=True)


    # === PARAMETERVERTEILUNGEN DER TOP-LÄUFE ===
    n_params = len(param_cols)
    cols_per_row = 5
    n_rows = math.ceil(n_params / cols_per_row)

    plt.figure(figsize=(cols_per_row * 3, n_rows * 3))
    for i, param in enumerate(param_cols):
        plt.subplot(n_rows, cols_per_row, i + 1)
        sns.histplot(df_top[param], kde=True)
        plt.title(param)
    plt.tight_layout()
    plt.suptitle(f"Parameterverteilungen (Top {int(percentile_threshold*100)}%)", y=1.02)
    param_plot_path = os.path.join( project.output_dir, f"parameters_{like_type}.png")
    plt.savefig(param_plot_path, dpi=300)
    plt.close()


    # === BESTE SIMULATION UND TOP-SIMULATIONEN ===
    sim_array = df_top[sim_cols].to_numpy()
    best_sim = df_sorted.iloc[0][sim_cols].to_numpy()
    n_steps = sim_array.shape[1]

    # Unsicherheitsbalken berechnen (z. B. 5.–95. Perzentil)
    lower = np.percentile(sim_array, 5, axis=0)
    upper = np.percentile(sim_array, 95, axis=0)
    error = [np.maximum(0.0, best_sim - lower), np.maximum( 0.0, upper - best_sim)]  # asymmetrisch
    #error = best_sim
    # === SCATTERPLOT MIT FEHLERBALKEN ===

    best_like = df_sorted.iloc[0][like_type]
    min_val = min(observed_values.min(), best_sim.min())
    max_val = max(observed_values.max(), best_sim.max())



    import matplotlib.gridspec as gridspec

    # Get parameter values for the best simulation
    best_params = df_sorted.iloc[0][param_cols]

    # Set up figure with GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # plot on left, table on right

    # === SCATTER PLOT ===
    ax0 = fig.add_subplot(gs[0])
    ax0.errorbar(observed_values, best_sim, yerr=error, fmt='o', ecolor='lightblue', alpha=0.6,
                 label=f'Unsicherheitsband (Top {int(percentile_threshold*100)}%)')
    ax0.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Linie')
    ax0.scatter(observed_values, best_sim, color='blue',
                label=f'Beste Simulation (like1 = {best_like:.3f})')

    ax0.set_xlabel("Beobachtete Werte")
    ax0.set_ylabel("Simulierte Werte")
    ax0.set_title("Beste Simulation mit Unsicherheitsband")
    ax0.set_xlim(0, 1.1 * max_val)
    ax0.set_ylim(0, 1.1 * max_val)
    ax0.set_aspect('equal', adjustable='box')
    ax0.legend()

    # === PARAMETER TABLE ===
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')  # turn off the axis

    # Prepare table data
    table_data = [[param, f"{value:.4g}"] for param, value in best_params.items()]

    table = ax1.table(cellText=table_data, colLabels=["Parameter", "Wert"], loc='center', cellLoc='left')
    table.scale(3, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        if col == 1:
            cell.set_width(0.5)  # Adjust for better spacing (default ~0.05)
            cell.set_text_props(ha='left', va='center')  # Better text alignment

    # Save combined figure
    scatter_plot_path = os.path.join( project.output_dir, f"opt_{like_type}_with_table.png")
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()

    return
    if self._setting.outputformat == 'csv':

        par_list = [p.name for p in self.params]

        #get data
        results = np.transpose( _results)

        sorted_indices = np.argsort(results[0])[::-1]
        results = results[:, sorted_indices]

        df_par = results[:len(par_list)+1]
        df_par = pd.DataFrame({'likelihood': df_par[0]})
        for n,c in zip(par_list, results[1:len(par_list)+1]):
            df_par[n] = c

        df_par.to_csv( self._setting.output+"_like.csv", index=False)

        results = results[len(par_list)+1:-1]
        results = pd.DataFrame(results, columns=["TEMP_"+str(i+1) for i in range(len(results[0]))])
        results['datetime'] = self._evaluation.index.strftime('%Y-%m-%d 00:%M:%S')

        #add time from evaluation data and write file
        self._evaluation['index'] = np.arange( len(self._evaluation))
        for c in self._setting.calibrations:
            ev_index = self._evaluation['index'].loc[self._evaluation[c['id']].notna()]
            header = [h.replace('TEMP', c['id']) for h in results.columns]

            data_out = results.iloc[ev_index]
            data_out.columns = header
            data_out["default"] = self._simulation_default[c['id']].dropna().values
            data_out.to_csv(f"{self._setting.output}_{c['id']}.kkplot", sep="\t", index=False)

        fig, ax = plt.subplots(nrows=math.ceil(5/3), ncols=3, figsize=(10, 4))
        nd_bins = 10

        # Filter the DataFrame to get the top 5% largest values
        df = df_par[df_par['likelihood'] >= df_par['likelihood'].quantile(0.9)]
        #df = df_par.nlargest(50, ['likelihood'])
        for par, x in zip(par_list, ax.flat):

            df.hist(column=par, bins=nd_bins, grid=True, color='#86bf91', zorder=2, rwidth=0.9, ax=x)

            # Despine
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
            x.spines['left'].set_visible(False)

            # Switch off ticks
            x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="off")

            x.set_title("")

            # Set x-axis label
            x.set_xlabel(par[3:]) #, labelpad=20, weight='bold', size=12)

            #x.set_xlim(0,1)
            x.set_yticklabels([])
        fig.savefig("histogram.png")

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
        ### Spotpy setup
        setup = spot_setup( config, project)

        lspotpy_functions = dict( { 'lhs': spotpy.algorithms.lhs,
                                    'fast': spotpy.algorithms.fast,
                                    'mcmc': spotpy.algorithms.mcmc})

        if project.parallel:
            sampler = lspotpy_functions[setup.method]( setup,
                                                       dbname=project.setting.output,
                                                       dbformat=project.setting.outputformat,
                                                       parallel = 'mpi')
        else:
            sampler = lspotpy_functions[setup.method]( setup,
                                                       dbname=project.setting.output,
                                                       dbformat=project.setting.outputformat)

        sampler.sample( setup.repetitions)

        setup.finalize( sampler)

    postprocess( project)

if __name__ == '__main__' :

    main()






