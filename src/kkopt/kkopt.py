#!/bin/bash

import string
import os
import sys
from os.path import exists
from dotenv import load_dotenv
import time
import re
import pandas as pd
import numpy as np
import math
from kkplot.kkutils.expand import *
from kkplot.kkutils.log import *
import kkopt.kkutils as utils
import numpy
import spotpy
from kkopt.kkopt_project import kkopt_project 
import numexpr as numexpr
from kkplot.kksources import kkplot_sourcefactory as kkplot_sourcefactory
from kkplot.kkplot_dviplot import *
from kkplot.kkplot_figure import DSSEP
import matplotlib.pyplot as plt
import subprocess

try:
    from mpi4py import MPI
except ImportError:
    raise Exception("MPI python module mpi4py not available. Exit!")


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

        #self._calibrations = _project.calibrations
        self.likes = []

        #objectivefunction
        self.objective_function = self._setting.get_property( 'likelihood')

        #run test simulation to get test output needed for 
        #preparation of evaluation data (maybe not needed or implement on demand)
        #utils.kklog.log_info('Start test simulation')
        self.update_parameters( None)
        self.run_simulation()

        #prepare evaluation data
        temp = self.get_data( 'simulation')
        self._evaluation = self.get_data( 'evaluation', temp)
        self._simulation = self.get_data( 'simulation', self._evaluation)
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
            eval_data = eval(expression).to_frame()
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

    def run_simulation( self) :

        models = []
        if 'submodels' in self._setting.properties['model']:
            for submodel in self._setting.properties['model']['submodels']:
                program = os.path.expandvars( submodel['binary'])
                argument = ''
                for arg in submodel['arguments'] :
                    argument = argument + os.path.expandvars( arg) + ' '
                models.append( program+" "+argument + " > /dev/null 2>&1")
        else:
            program = os.path.expandvars( self._setting.properties['model']['binary'])
            argument = ''
            for arg in self._setting.properties['model']['arguments'] :
                argument = argument + os.path.expandvars( arg) + ' '
            models.append( program+" "+argument + str(MPI.COMM_WORLD.Get_rank()+2))# + " > /dev/null 2>&1")

        t0 = time.time()

        kklog_info('Model run %s' %models[-1])
        kklog_info('Rank %s' %str( MPI.COMM_WORLD.Get_rank()+2))

        if True: #self._setting.properties['model']['mode'] == 'parallel':

            # List to store subprocess objects
            processes = []

            # Start each command as a subprocess
            for m in models:
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
        Lresources_path = os.path.expanduser(kkexpand('${HOME}')+'/.ldndc/Lresources_tmp_'+str(MPI.COMM_WORLD.Get_rank()+2))
        if not os.path.exists( Lresources_path):
            os.makedirs( Lresources_path)
        if not os.path.exists( Lresources_path+"/udunits2"):
            shutil.copytree( kkexpand('${HOME}')+'/.ldndc/udunits2', Lresources_path+'/udunits2')
        with open(os.path.join( Lresources_path, 'Lresources'), 'w') as f:
            f.write( subject)

    def simulation( self, _parameters=None) :

        if _parameters is not None:
            self.update_parameters( _parameters)
 
        (rc,time) = self.run_simulation()
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

        results = _sampler.getdata()

        #try:
        #    spotpy.analyser.plot_fast_sensitivity( results, number_of_sensitiv_pars=3)
        #except:
        #    pass

        #self.postprocess( np.asarray([np.asarray(item) for item in results.astype(object)]))

    def postprocess( self, _results):
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

    kkplot_env = kkexpand( '${HOME}')+'/.kkplot/kkplot.env'
    if ( exists( kkplot_env)) :
        load_dotenv( kkplot_env)
    kkplot_env = kkexpand( '${HOME}')+'/.ldndc/kkplot.env'
    if ( exists( kkplot_env)) :
        load_dotenv( kkplot_env)

    ### Configuration object
    config = utils.configuration()

    ### Calibration project
    project = kkopt_project( config)

    ### Spotpy setup
    setup = spot_setup( config, project)

    lspotpy_functions = dict( { 'lhs': spotpy.algorithms.lhs, 'fast': spotpy.algorithms.fast,
                                'mcmc': spotpy.algorithms.mcmc})
    sampler = lspotpy_functions[setup.method]( setup,
                                               dbname=project.setting.output, 
                                               dbformat=project.setting.outputformat, 
                                               parallel = 'mpi')
    sampler.sample( setup.repetitions)

    setup.finalize( sampler)

if __name__ == '__main__' :

    main()
