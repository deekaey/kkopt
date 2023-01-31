
#from kkutils.log import *
from kkplot.kkutils.expand import *
from kkplot.kkplot_figure import kkplot_datasource
from kkplot.kkplot_figure import kkplot_namedconstants
from kkplot.kkplot_figure import kkplot_namedfunctions
from kkplot.kkplot_figure import KKCONSTS
from kkplot.kkplot_figure import KKFUNCS
import yaml

## global named constants dictionary
kkplot_defines = kkplot_namedconstants( KKCONSTS)
## global named functions dictionary
kkplot_functions = kkplot_namedfunctions( KKFUNCS)




class kkopt_setting( object) :
    def __init__( self) :

        self._model = { 'program': None, 'arguments': None }

        self._properties = dict()
        self._parameters = dict()
        self._evaluations = []
        self.datasource = kkplot_datasource( ':none:')

        self._output = 'output'

        self.calibrations = []


    def add_parameter_file( self, _parameter_file) :
        pf_stream = open( _parameter_file, 'r')
        content = yaml.load( pf_stream, Loader=yaml.FullLoader)
        for k,v in content['parameters'].items():
            self._parameters.update( {k: v})

    def add_parameter( self, _parameters) :
        self._parameters.update( _parameters)

    def add_model( self, _model) :
        self._model.update( _model)

    def add_property( self, _key, _value) :
        self._properties.update( { _key: _value })

    def add_evaluations( self, _evaluations) :
        self._evaluations.append( _evaluations)

    def add_sources( self, _sources) :
        self._datasources.update( _sources)



    def set_output( self, _output) :
        self._output = _output

    @property
    def datasources( self) :
        return self._datasources

    @property
    def properties( self) :
        return self._properties

    def get_property( self, _property, _default=None) :
        return self._properties.get( _property, _default)

    @property
    def repetitions( self) :
        return self._properties['repititions']

    @property
    def simulationtime( self) :
        return self._simulationtime

    @property
    def output( self) :
        return ''.join((self._properties['output']).split('.')[:-1])

    @property
    def outputformat( self) :
        return (self._properties['output']).split('.')[-1]

    @property
    def evaluations( self) :
        return self._evaluations

    @property
    def parameters( self) :
        return self._parameters

    def __str__( self) :
        return str(self._properties) 

#        return 'title=%s; calibrations=%s' \
#            % ( self._title, ','.join( [ str( self._calibrations[cal]) for cal in self._calibrations]))
#
#
#
#    ## exposed constants
#    @property
#    def groupbytag( self) :
#        return GROUPBYTAG
#    @property
#    def namespaceseparator( self) :
#        return NSSEP
#    @property
#    def datasourceseparator( self) :
#        return DSSEP
#
    def add_defines( self, _defines) :
        kkplot_defines.add_constants( _defines)

