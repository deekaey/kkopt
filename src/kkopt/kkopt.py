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

        self._calibrations = _project.calibrations

        #objectivefunction
        self.objective_function = self._setting.get_property( 'likelihood')
        
        #run test simulation to get test output needed for 
        #preparation of evaluation data (maybe not needed or implement on demand)
        #utils.kklog.log_info('Start test simulation')
        self.run_simulation()

        #prepare evaluation data
        simulation_data = self.get_data( 'simulation')
        self._evaluation = self.get_data( 'evaluation', simulation_data.index)

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

    def _scan_expression( self, _expression) :
        digits = '0123456789'
        identifierchars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_' + digits + NSSEP + DSSEP
        identifiernonheadchars = digits + DSSEP
        operatorchars = '+-*/%'
        identifiers = list()
        functions = list()
        namedconstants = list()
        numbers = list()
        tokens = list()

        token = ''
        number = ''
        expression = '%s%s' % ( _expression, '$') ## add stop marker
        for ( p, c) in enumerate( expression) :
            if token == '' and ( c in digits or ( number != '' and ( c == '.' or c == 'e' or c == 'E'))) :
                number += c
            elif c in identifierchars :
                if token == '' :
                    identpos = p
                token += c
            elif c == '(' and token != '' :
                functions.append( token) #kkterm( p, token))
                tokens.append( token)
                token = ''
            else :
                if token != '' and token != NSSEP :
                    if token[0] in identifiernonheadchars :
                        kklog_fatal( 'invalid identifier')
                        raise RuntimeError( 'invalid identifier')
                    if False :#self._isnamedconst( token) :
                        namedconstants.append( token) #kkterm( p, token))
                        tokens.append( token)
                    else :
                        identifiers.append( token)
                        tokens.append( token)
                    token = ''
                elif number != '' :
                    numbers.append( number)
                    tokens.append( number)
                    number = ''
                if c in operatorchars and token != '$' :
                    tokens.append( c)
            if c in '(),' :
                tokens.append( c)

        return  dict( I=identifiers, F=functions, D=namedconstants, T=tokens)

    def _resolve_defines( self, _expression) :
        expr = self._scan_expression( _expression)
        tokens = expr["T"]
        namedconstants = expr["D"]
        functions = expr["F"]

        expression = ""
        for token in tokens :
            tokenreplace = token
            #if token in namedconstants :
            #    if not self._isnamedconst( token) :
            #        kklog_fatal( 'named constant not defined  [token=%s]' % ( token))
            #    tokenreplace = '(%s)' % ( str( self._getconst( token)))
            if token in functions :
                #if not self._isfunc( token) :
                #    kklog_fatal( 'function not defined  [token=%s]' % ( token))
                tokenreplace = str( self._getfunc( token))
            expression += tokenreplace
        return expression

    def _getconst( self, _token) :
        return kkopt_defines.value( _token)

    def _isfunc( self, _token) :
        return _token in kkopt_functions.functions

    def _getfunc( self, _token) :
        return kkopt_functions.name( _token)

    def get_data( self, _target, _index=None) :

        datasource_name = self._setting.calibrations[0][_target]['datasource'].name
        sampletime = self._setting.calibrations[0]['sampletime']
        entity = self._setting.calibrations[0][_target]['entity']
        path = self._setting.calibrations[0][_target]['datasource'].path
        data = pd.read_csv( path, header=0, na_values=['-99.99','na','nan'], comment='#', sep="\t")
        data = self.canonicalize_headernames( data)

        t_from, t_to = sampletime.split( '->')
        eval_data = data.loc[(data['datetime'] >= t_from) & (data['datetime'] <= t_to),]
        eval_data = eval_data.set_index('datetime')
        eval_data.index = pd.to_datetime(eval_data.index)

        if 'filter' in self._setting.calibrations[0][_target]:
            for f in self._setting.calibrations[0][_target]['filter']:
                for k,v in f.items():
                    eval_data = eval_data.loc[eval_data[k].isin(v),]
        eval_data = eval_data[[entity]]

        expression = self._setting.calibrations[0][_target]['expression']
        expression = expression.replace( entity+DSSEP+datasource_name, 'eval_data["%s"]' %entity)
        eval_data = eval(expression).to_frame()
        if _index is not None:
            eval_data = eval_data.loc[eval_data.index.isin(_index),:]
        return eval_data

    @property
    def dbname( self) :
        return self._setting.output

    @property
    def repetitions( self) :
        return self._setting.repetitions

    # This function is needed for spotpy to compare simulation and validation data
    # Keep in mind, that you reduce your simulation data to the values that can be compared with observation data
    # This can be done in the def simulation (than only those simulations are saved), 
    # or in the def objectivefunctions (than all simulations are saved)
    def objectivefunction( self, simulation, evaluation) :
        L = -99.99
        if self.objective_function == 'r2' :
            L = spotpy.objectivefunctions.rsquared( evaluation, simulation)
        elif self.objective_function == 'rmse' :
            L = -spotpy.objectivefunctions.rmse( evaluation, simulation)
        return L

    def run_simulation( self) :

        program = os.path.expandvars( self._setting.properties['model']['binary'])
        argument = ''
        for arg in self._setting.properties['model']['arguments'] :
            argument = argument + os.path.expandvars( arg) + ' ' 

        t0 = time.time()
        os.system( program+' '+argument + "> /dev/null 2>&1")
        t1 = time.time()

        return round( (t1-t0),2)

    def simulation( self, _parameters=None) :

        if _parameters is not None:
            # open the source file and read it
            subject = ''
            with open('/Users/kraus-d/.ldndc/Lresources', 'r') as f:
                subject = f.read()
            
            p_index = 0
            for k,v in self._setting.parameters.items() :
                search = re.search(r'.*\.%s\..*' % v['name'], subject)
                pattern = re.compile(r'.*\.%s\..*' % v['name'])
                subject = pattern.sub('%s = "%f"' %(search.group(0).split('=')[0], _parameters[p_index]), subject)            
                p_index += 1

            # write the file
            with open('/Users/kraus-d/.ldndc/Lresources', 'w') as f:
                f.write(subject)
 
        time = self.run_simulation()
        kklog_debug('Simulation duration %s s' %str(time))

        results = self.get_data( 'simulation', self._evaluation.index)
        if None in results :
            kklog_fatal("loading of simulation data failed")
            results = [ [numpy.nan]*len(series) for series in self.O ]

        return results.squeeze().values

    def evaluation( self):
        return self._evaluation.squeeze().values

    #write spoty output more userfriendly
    def finalize( self) :

        if self._setting.outputformat == 'csv':
            infile = open( self._setting.output+".csv", 'r')
            header_line = infile.readline()
            infile.close()

            #get parameter list
            header_line = header_line.split(',')
            par_list = [par for par in header_line if 'par' in par]

            #get data
            data = np.genfromtxt( self._setting.output+".csv", skip_header=1, delimiter=',')
            data = np.transpose(data)
            header_likelihood = data[0]
            header_rang = [self.objective_function+"_"+str(i[0]+1) for i in sorted(enumerate(header_likelihood), key=lambda x:x[1], reverse=True)]
            header_out = [-99.99 for i in range(len(header_likelihood))]
            #remove lines that do not contain data and add header
            data = data[len(par_list)+1:-1]
            data = pd.DataFrame(data, columns=header_rang)
            #add time from evaluation data and write file
            eval_data = self.get_data( 'evaluation')
            data['datetime'] = eval_data.index.strftime('%Y-%m-%d 00:%M:%S')
            data.to_csv(self._setting.output+".kkplot", sep="\t", index=False)

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
    sampler = spotpy.algorithms.lhs( setup,
                                     dbname=project.setting.output, 
                                     dbformat=project.setting.outputformat, 
                                     parallel='seq')

    sampler.sample( setup.repetitions)

    setup.finalize()

if __name__ == '__main__' :

    main()
