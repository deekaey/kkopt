#!/bin/bash

"""
Created on Wed Mar 11 14:24:36 2015

@author: houska-t
"""
import os
import sys
import time
import datetime
import re

import numpy
import spotpy

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
__lspotpy_interpolation_factory_linear = lspotpy_interpolation_factory_linear( 'linear')


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


class data_loader( object) :
    def  __init__( self, _begin, _end) :
        self.data = dict()
        self.begin = _begin
        self.end = _end

    def  load( self, _file, _key, _column=2) :
#        try :
        data = numpy.genfromtxt( _file, usecols=( _column), skip_header=1)
#        except 
#           data = None
## TODO  crop observation data to simulation time period
        #data = data[numpy.where( data['julianday'])>=self.begin...

        self.data[_key] = data
        return  data

    def  __getitem__( self, _key) :
        return  self.data[_key]

    def __iter__( self) :
        for key in self.data :
            yield  key

class spot_setup(object):
    def __init__(self, _basepath=None, _model='', _parallel='seq'):

        self.basepath = _basepath if _basepath is not None else os.getcwd()
        self.basepath += os.sep

        self.model = _model

        self.analysestart = datetime.datetime( 2004, 1, 1) #Model starts
        self.datastart = datetime.datetime( 2006, 10, 6) #Comparision with Eval data starts
        self.dataend = datetime.datetime( 2007, 10, 6)

        self.project_path = self.basepath + os.sep
        self.input_path = self.basepath + os.sep
        self.output_path = self.basepath + os.sep
        self.observation_path = self.basepath + os.sep

## FIXME  potential ordering problem for multiple entries
        self.observation_files = { 'soilwater_10cm':'soilwater10cm-observed.txt'}
        self.O = self.load_data( self.observation_files, self.observation_path)
        self.result_files = { 'soilwater_10cm':'soilwater10cm-simulated.txt'}

        dtype=[ ('source','S32'), ('module', 'S40'), ('name', 'S40'), ('distribution', 'S40'), ('init', '<f8'), ('min', '<f8'), ('max', '<f8')]
        self.P = numpy.genfromtxt( self.project_path + 'parameters.txt', names=True, dtype=dtype)
        self.n_P = 10

        self.sources = list( numpy.unique( self.P['source']))

    def load_data( self, _datasource, _datapath):
        dataloader = data_loader( self.datastart, self.dataend)
        for key in _datasource :
            dataloader.load(  _datapath+_datasource[key], key)
        return  dataloader

    def parameters( self):
        pars = []   #distribution of random value      #name  #stepsize# optguess
        i = 0
## TODO  alternatively, sort them by source!
        for source in self.sources :
            source_pars = self.P[self.P['source']==source]
            for par in source_pars :
                pars.append(( numpy.random.uniform( low=self.P['min'][i], high=self.P['max'][i]),
                    self.P['name'][i], ( self.P['max'][i]-self.P['min'][i]) / float( self.n_P), self.P['init'][i]))
                i += 1

        dtype = numpy.dtype([('random', '<f8'), ('name', '|S30'),('step', '<f8'),('optguess', '<f8')])
        return numpy.array( pars, dtype=dtype)
    
    def get_results( self) :
        results = self.load_data( self.result_files, self.output_path)
        return  [ results[series] for series in results ]

    def simulation( self, _parameters) :
#        try:        
#            call = int(os.environ['OMPI_COMM_WORLD_RANK'])
#        except KeyError:
#            call=int(numpy.random.uniform(low=0,high=999999999))
#        self.write_parameters(vector,call=call)
        print 'Start model...'
        t0 = time.time()
#        os.chdir(self.owd+os.sep+self.version)
#        path="projects"+os.sep+"grassland"+os.sep+"DE_linden"+os.sep+"DE_linden_"+self.module         
#        os.system( r"bin"+os.sep+"ldndc -c ldndc.conf "+path+os.sep+"DE_linden_"+self.module+str(call)+".xml")
        #print _parameters
        t1 = time.time()
        print 'Duration: ' + str( round((t1-t0),2)) + ' s'

        results = self.get_results()
        if None in results :
            sys.stderr.write( '[EE] loading of simulation data failed\n')
            #print str(sys.exc_info()[1])
            results = [ [numpy.nan]*len(series) for series in self.O ]

        return  results

    def evaluation( self, return_dates=False):
        return [ self.O[series] for series in self.O ]

    def likelihood( self, _simulation, _observation):
        L = -spotpy.likelihoods.rmse( _simulation, _observation)
        return  L


if __name__ == '__main__' :

    rep=1
    # Check if script is started with mpirun - works only with OpenMPI
    parallel = 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'

#    path=''
#    if parallel == 'mpi' :
#        path = os.sep+'fhgfs'+os.sep+'gh2365'+os.sep
    spot_setup = spot_setup( _basepath='examples', _parallel=parallel)

    sampler = spotpy.algorithms.lhs( spot_setup,
        dbname=spot_setup.project_path+'LHS_'+spot_setup.model, dbformat='csv', parallel=parallel, save_sim=True)
    sampler.sample( rep)

