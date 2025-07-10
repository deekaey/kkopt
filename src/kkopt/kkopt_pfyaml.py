
#from kkopt.kkutils.expand import *
from kkopt.kkopt_setting import *
from kkplot.kkutils.expand import *
from kkplot.kkplot_figure import kkplot_datasource
from kkplot.kkplot_figure import kkplot_expressions
from kkplot.kkplot_figure import DSSEP
from kkplot.kkplot_pfyaml import load as kkopt_load
from kkplot.kkplot_pfyaml import merge_plotfiles
import kkplot.kkplot_provider as kkplot_provider
import yaml

class kkopt_pfreader_yaml( object) :
    def __init__( self, _conf) :
        self._conf = _conf
        self._setting = kkopt_setting()
        self._pf_name = _conf.optfile()
        self._pf_data = None

        rc_load, self._pf_data = kkopt_load( self._pf_name, self._pf_data)
        if rc_load :
            raise RuntimeError( 'loading config file failed')
        kklog_debug( 'loading calibration configuration successful')

        rc_read = self.read()
        if rc_read :
            raise RuntimeError( 'reading config file failed')
        kklog_debug( 'reading calibration configuration successful')

    @property
    def calibrationfile( self) :
        return  self._pf_name

    def read( self) :
        kklog_debug( 'reading configuration [%s]' % self._pf_name)

        self._setting = self.read_setting()
        if self._setting is None :
            return -1

        if self.read_namedconstants() :
            return -1

        if self.read_calibrations() :
            return -1

        return  0

    def read_setting( self) :

        setting = kkopt_setting()

        config = self._pf_data.get( 'configuration')
        if config :
            calibration_title = config.get( 'title', '')
            calibration_title = kkexpand( calibration_title)

            calibration_output = config.get( 'output', None)
            calibration_output = kkexpand( calibration_output)

            for i in ['simulationtime', 'sampletime', 
                      'repititions', 'model',
                      'method', 'likelihood', 
                      'parameterdistribution', 'parameters', 
                      'output', 'outputformat', 
                      'parallel']:
                if self._is_valid( config, i) :
                    if i == 'parameters':
                        setting.add_parameter_file( kkexpand( config[i]))
                    elif type(config[i]) == str:
                        setting.add_property( i, kkexpand( config[i]))
                    else:
                        setting.add_property( i, config[i])
        else :
            return None

        if self._is_valid( config, 'datasource') :
            setting.datasource = self.read_source( config['datasource'], kkplot_datasource())
        #sources = self._pf_data.get( 'datasources')
        #self._setting.add_sources( sources)

        return setting

    def read_namedconstants( self) :
        if self._is_valid( self._pf_data, 'define') :
            self._setting.add_defines( self._pf_data['define'])
        return 0

    def read_calibration_infos( self, _node, _defaults) :
        plot_source = _defaults['datasource']
        if _node and 'datasource' in _node.keys() :
            plot_source = self.read_source( \
                _node['datasource'], _defaults['datasource'])
        return { 'datasource':plot_source }

    def read_calibrations( self) :
        if len( self._pf_data.get( 'calibrations', [] )) == 0 :
            return 0
 
        calibration_info_defaults = { 'datasource': self._setting.datasource }
        for calibration_k in self._pf_data['calibrations'] :

            calibration_id = list(calibration_k.keys())[0]
            if not self._is_valid_id( "plotID", calibration_id) :
                return  -1

            calibration_block = calibration_k[calibration_id]

            calibration_infos = self.read_calibration_infos( calibration_block, calibration_info_defaults)

            #graph_info_defaults = { 'datasource':calibration_infos['datasource'] }

            calibration_datasource = calibration_infos['datasource']

            add_calibration = dict({'id': calibration_id})
            if 'sampletime' in calibration_block:
                add_calibration.update({'sampletime': calibration_block['sampletime'] })
            for i in ['evaluation', 'simulation'] :
                exprs = kkplot_expressions( calibration_id+'.'+i, [calibration_block[i]['name']])
                for terminal in exprs.terminals :
                    datasource = None
                    terminal_with_source = [ s.strip() for s in terminal.split( DSSEP)]
                    if len( terminal_with_source) == 1 :
                        datasource = calibration_datasource
                    elif len( terminal_with_source) == 2 :
                        kklog_debug( 'reading datasource information for terminal "%s"' % ( terminal))
                        datasource = self.read_source( \
                            terminal_with_source[1], calibration_datasource)
                    else :
                        kklog_error( 'invalid column specification  [column=%s]' % ( terminal))
                        return -1
                    if datasource is None :
                        return -1

                    add_entity = dict( {'expression': calibration_block[i]['name'], 
                                        'entity': terminal_with_source[0],
                                        'datasource': datasource} )
                    if 'filter' in calibration_block[i]:
                        add_entity.update( {filter: calibration_block[i]['filter']} )

                    add_calibration.update({i: add_entity}) 

            self._setting.calibrations.append( add_calibration)
        return  0

    def read_evaluations( self, _calibration_block) :
        if 'evaluation' in _calibration_block :
            return _calibration_block['evaluation']
        return

    def read_simulations( self, _calibration_block) :
        if 'simulation' in _calibration_block :
            return _calibration_block['simulation']
        return

    def read_calibration_infos( self, _node, _defaults) :
        calibration_source = _defaults['datasource']
        if _node and 'datasource' in _node.keys() :
            calibration_source = self.read_source( \
                _node['datasource'], _defaults['datasource'])
        return { 'datasource':calibration_source }

    def read_source( self, _source, _refsource) :
        if not _source :
            return  kkplot_datasource( '%s' % ( _refsource.name))
        if type( _source) is dict :
            return  self._read_source( _source, _refsource)
        elif type( _source) is str :
            sources = self._pf_data['datasources']
            if not sources :
                return  None
            source = sources.get( _source)
            if not source :
                kklog_error( 'no such source block  [datasource=%s]' % ( _source))
                return  None

            return  self._read_source( source, _refsource)

        else :
            kklog_error( 'i do not know what kind of datasource i am looking at')
            return  None

    def _read_source( self, _source, _refsource) :
        new_source = kkplot_datasource( '%s' % ( self._get_source_name( _source)))

        if self._is_valid( _source, 'kind') :
            new_source.kind = _source['kind']
        else :
            new_source.kind = _refsource.kind

        if self._is_valid( _source, 'format') :
            new_source.format = _source['format']
        else :
            new_source.format = _refsource.format
        new_source.add_formatargs( _refsource.formatargs)
        new_source.add_formatargs( _source.get( 'formatargs'))

        if self._is_valid( _source, 'flavor') :
            new_source.flavor = _source['flavor']
        else :
            new_source.flavor = _refsource.flavor
        new_source.add_flavorargs( _refsource.flavorargs)
        new_source.add_flavorargs( _source.get( 'flavorargs'))

        if self._is_valid( _source, 'path') :
            new_source.set_path( _source['path'], self._conf.base_dir_for( new_source.kind))
        else :
            new_source.set_path( _refsource.path)

        if self._is_valid( _source, 'provider') :
            s_provider = _source['provider']
            if self._is_valid( s_provider, 'program') :
                p_program = s_provider['program']
            else :
                p_program = None
            if self._is_valid( s_provider, 'arguments') :
                p_arguments = s_provider['arguments']
            else :
                p_arguments = None
            new_provider = kkplot_provider.kkplot_provider( \
                p_program, p_arguments, self._conf.base_dir_for( 'providers'))
            new_source.provider = new_provider

        kklog_debug( new_source)
        return  new_source

    def _get_source_name( self, _source) :
        source_id = id( _source)
        sources = self._pf_data['datasources']
        if not sources :
            return source_id
        for source in sources :
            if id( sources[source]) == source_id :
                return source
        return ':%s:' % ( str( source_id))

    def _is_valid( self, _node, _tag) :
        return _node and _node.get( _tag) is not None

    def _is_valid_id( self, _id_kind, _id) :
        validchars_ID = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
        invalid_chars = list()
        nb_invalid_chars = 0
        for c in _id :
            if c not in validchars_ID :
                nb_invalid_chars += 1
                if c not in invalid_chars :
                    invalid_chars.append( c)
        if len( invalid_chars) != 0 :
            kklog_error( '%s "%s" contains %d invalid character(s): %s' \
                % ( _id_kind, _id, nb_invalid_chars, ''.join(invalid_chars)))
            return  False
        return  True
