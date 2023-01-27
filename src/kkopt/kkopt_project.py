
#from kkutils.log import *
#from kkutils.timedelta import parse_timedelta
from kkopt.kkopt_setting import *
from kkopt.kkopt_pfyaml import kkopt_pfreader_yaml as pfreader_yaml
#from kkplot_figure import kkplot_figure as kkplot_figure
#from kkplot_figure import isreference as kkplot_isreference
#from kkplot_figure import asname as kkplot_asname
#from kkplot_figure import nocolumndepends as kkplot_nocolumndepends

#from kksources import kkplot_sourcefactory as kkplot_sourcefactory

import os as os
import pandas as pandas
import numpy as numpy
import numexpr as numexpr


class kkopt_project( object) :
    def __init__( self, _conf, _pf_format='yaml') :

        self._conf = _conf
        self._setting = kkopt_setting()
        if _pf_format == 'yaml' :
            pf_reader = pfreader_yaml( _conf)
            self._setting = pf_reader._setting

    @property
    def setting( self) :
        return self._setting

    def calibrations( self) :
        return self._setting.calibrations

    def properties( self) :
        return self._setting.properties

    @property
    def series_kinds( self) :
        return  [ 'time', 'space', 'non']

    @property
    def plotfile( self) :
        return  self._pf_name


    def get_property( self, _property, _default=None) :
        return self._figure.get_property( _property, _default)

    @property
    def groupbytag( self) :
        return self._figure.groupbytag



    @property
    def title( self) :
        return self._figure.title
    @property
    def outputfile( self) :
        of = self._outputfile
        if not os.path.isabs( of) :
            of = '%s/%s' % ( self._conf.outputs_dir(), of)
        if self._conf.bundle :
            of = of if of.rfind( os.sep) == -1 else of[of.rfind( os.sep)+1:]
        return  of
    @property
    def outputfileformat( self) :
        return self._outputfileformat

    @property
    def output_basename( self) :
        of = self._outputfile
        of = of if of.rfind( os.sep) == -1 else of[of.rfind( os.sep)+1:]
        of = of if of.rfind( '.') == -1 else of[:of.rfind( '.')]
        return  of

    @property
    def output_directory( self) :
        if self._conf.bundle :
            output_dir = self.output_basename
            output_dir = '%s/%s.dir/' % ( self._conf.outputs_dir(), output_dir)
        else :
            output_dir = '%s' % ( self._conf.tmp_dir())
        return  output_dir

    def create_output_directory( self) :
        output_dir = self.output_directory
        if not os.path.exists( output_dir) :
            os.makedirs( output_dir)
            if not os.path.exists( output_dir) :
                return  None
        return  output_dir

    def datapool_filename( self, _serieid, _abs=False) :
        fn = '%s-%s.%s' % ( self.output_basename, \
                canonicalize_name( _serieid), 'csv')
        if _abs or not self._conf.bundle :
            output_dir = self.output_directory
            fn = '%s/%s' % ( output_dir, fn)
        return fn

    def source_filename( self, _suffix) :
        fp = self.output_directory
        fn = self.output_basename
        return '%s/%s.%s' % ( fp, fn, _suffix)

    def __iter__( self) :
        for graph in self._toposorted_graphs :
            yield ( graph, graph._plot)
    @property
    def plots( self) :
        return [ plot for plot in self._figure ]

    def _construct_entity_readers( self, _entities) :
        entity_readermap = dict()
        print( "in entity")
        print( _entities)
        for entity in _entities :
            print( entity)
            graph = self._figure.get_graph( entity)


            for dependency in graph.dependencies( entity) :
                if not kkplot_isreference( dependency) :
                    datasource = graph.get_datasource( dependency)
                    kklog_debug( '  source=%s' % ( datasource.path))
                    kklog_debug( '    source me "%s"' % ( dependency))
                    if datasource.name in entity_readermap :
                        kklog_debug( 'already loaded source [%s]' % ( datasource.name))
                        continue
                    entity_reader = self._construct_entity_reader( dependency, datasource)
                    if entity_reader is None :
                        raise RuntimeError( '')
                    entity_readermap[datasource.name] = entity_reader

        print( "end entity")
        return entity_readermap

    def _construct_entity_reader( self, _entity, _datasource) :
        if _datasource is None or _datasource.path is None :
            raise RuntimeError( 'data source not set for entity "%s"' % ( _entity))

        entity_reader_constructor = kkplot_sourcefactory( _datasource, self._conf)
        kklog_debug( 'dviplot datasource flavor "%s" (%s)' % ( _datasource.flavor, _datasource.flavorargs))
        entity_reader = entity_reader_constructor.construct()
        if entity_reader is None :
            raise IOError( 'failed to read data source  [%s]' % ( _datasource.name))
        kklog_debug( 'loaded source [%s]' % ( _datasource.name))
        return entity_reader

    def _merge_data( self, _entities, _entity_readermap) :
        series = dict()
        for entity in _entities :
            graph = self._figure.get_graph( entity)
            if series.get( graph.graphid) is None :
                series[graph.graphid] = pandas.DataFrame()

## sk:obs?            skip = False
## sk:obs?            if graph.name is not None :
## sk:obs?                for graph_name in graph.names :
## sk:obs?                    if not graph_name in data.columns :
## sk:obs?                        kklog_warn( 'invalid column name in name list [column=%s]' % ( graph_name))
## sk:obs?                        skip = True
## sk:obs?                        break
## sk:obs?            if skip :
## sk:obs?                continue

            for dependency in graph.dependencies( entity) :
                if kkplot_isreference( dependency) :
                    continue

                datasource = graph.get_datasource( dependency)
                series_kind = graph.domainkind

                entity_reader = _entity_readermap[datasource.name]

                queryname = kkplot_asname( dependency)
                for dataselect in graph :
                    targetname = graph.dataid( dataselect, dependency)
                    if targetname in series[graph.graphid].columns :
                        kklog_debug( 'datacolumn "%s" for graph "%s" exists. skipping.' % ( targetname, graph.graphid))
                        continue
        
                    selected_data = pandas.DataFrame()
                    if graph.name is None :
                        pass
                    else :
                        selected_data = entity_reader.read( graph, [queryname], [targetname], dataselect)
                        if selected_data is None :
                            return None

                    if selected_data.empty :
                        kklog_warn( 'no data for graph  [graph=%s,datalabel=%s]' % ( graph.graphid, graph.datalabel( dataselect)))
                    else :
                        series[graph.graphid] = self._merge_data_join_series( graph.graphid, \
                            series_kind, series[graph.graphid], selected_data, \
                            graph.domain.domaincolumns, datasource)

        return series

    def _merge_data_join_series( self, _graphid, _kind, _series_pool, _series, _index_columns, _datasource) :
        if _kind == 'time' :
            return self._merge_data_join_timeseries( _series_pool, _series, _index_columns, _datasource)
        elif _kind == 'space' :
            return self._merge_data_join_spaceseries( _series_pool, _series, _index_columns, _datasource)
        elif _kind == 'non' :
            return self._merge_data_join_nonseries( _series_pool, _series)
        else :
            kklog_fatal( 'unknown series kind [kind=%s]' % ( _kind))
        return None
    def _merge_data_join_nonseries( self, _nonseries_pool, _nonseries) :
        for column in _nonseries.columns :
            _nonseries_pool[column] = _nonseries[column]
        return _nonseries_pool
    def _merge_data_join_timeseries( self, _timeseries_pool, _timeseries, _index_columns, _datasource) :
        timeseries = self._transform_domaincolumns_time( _timeseries, _index_columns, _datasource)
        timeseries = timeseries.set_index( _index_columns)
        return pandas.concat( [_timeseries_pool, timeseries], axis=1, join='outer')
    def _merge_data_join_spaceseries( self, _spaceseries_pool, _spaceseries, _index_columns, _datasource) :
        spaceseries = _spaceseries.set_index( _index_columns)
        timeseries = self._transform_domaincolumns_space( timeseries, _datasource)
        return pandas.concat( [_spaceseries_pool, spaceseries], axis=1, join='outer')

    def _transform_domaincolumns_time( self, _timeseries, _index_columns, _datasource) :
        t_shiftarg = _datasource.flavorarg( 'shift')
        if t_shiftarg is not None :
            t_shift = parse_timedelta( t_shiftarg)
            if t_shift is not None :
                kklog_verbose( 'time-shifting data "%s" by [%s]' % ( _datasource.name, t_shift))
                _timeseries[_index_columns] += pandas.to_timedelta( t_shift)
        return  _timeseries
    def _transform_domaincolumns_space( self, _spaceseries, _datasource) :
        t_shiftarg = _datasource.flavorarg( 'shift')
        if t_shiftarg is not None :
            kklog_warn( 'missing implementation for space shift')
        return  _spaceseries


    def _evaluate_domain_expressions( self, _series, _entities) :
        entities = list([ entity for entity in _entities])
        series = dict()
        for entity in entities :
            #kklog_debug( 'entity="%s"' % ( entity))
            if kkplot_nocolumndepends( entity) :
                continue
            graph = self._figure.get_graph( entity)
            serie = _series.get( graph.graphid, None)
            if serie is None :
                kklog_fatal( 'missing data for entity  [entity=%s]' % ( entity))
            series[graph.graphid] = self._evaluate_expressions( _series, entity, graph)
        return series

    def _evaluate_expressions( self, _series, _entity, _graph) :
        for dataselect in _graph :
            entity_assign, expression = \
                self._rewrite_expression( _series, _entity, _graph, dataselect)
            kklog_info( '_series["%s"]["%s"] = %s' % ( _graph.graphid, entity_assign, expression))
            #_series[_graph.graphid][entity_assign] = eval( expression)
            a = pandas.DataFrame()
            a[entity_assign] = eval( expression)

            _series[_graph.graphid] = pandas.concat( [_series[_graph.graphid], a], axis=1, join='outer')

        return _series[_graph.graphid]

    def _rewrite_expression( self, _series, _entity, _graph, _dataselect) :
        expression = _graph.expression( _entity)
        #kklog_info( '%s = %s' % ( _graph.graphid, expression))
        dependencies = list( _graph.dependencies( _entity))
        ## sort by string length (descending) to disambiguate
        dependencies.sort( lambda d1, d2: cmp( len(d2), len(d1)))

        dataselect_index = _graph.asindex( _dataselect)
        if dataselect_index != '' :
            dataselect_index = '%s%s' % ( self.groupbytag, dataselect_index)

        entity_assign = '%s%s' % ( _entity, dataselect_index)
        #kklog_debug( 'entity_assign= "%s" + "%s"' % ( _entity, dataselect_index))

        rewrite_expression = expression
        for dependency in dependencies :
            dependency_name = _graph.dataid( _dataselect, dependency)
            graph = self._figure.get_graph( dependency_name)

            rewrite_expression = rewrite_expression.replace( \
                dependency, dependency_name)
            rewrite_expression = rewrite_expression.replace( \
                dependency_name, '%s["%s"]["%s"]' % ( '_series', graph.graphid, dependency_name))

        #kklog_info( '%s = %s\n' % ( _graph.graphid, rewrite_expression))
        return ( entity_assign, rewrite_expression)

    def _delete_terminal_columns( self, _series) :
        for serieid in _series :
            serie = _series.get( serieid, None)
            if serie is None :
                continue
            terminal_columns = [ terminal_column for terminal_column in serie.columns \
                if terminal_column.startswith( self._figure.datasourceseparator) ]
            if len( terminal_columns) > 0 :
                kklog_verbose( 'dropping %d unused columns from %s data pool' % ( len( terminal_columns), serieid))
                serie.drop( terminal_columns, inplace=True, axis=1)

        return _series

    def _write_data( self, _series) :
        for serieid in _series :
            serie = _series.get( serieid, None)
            if serie is None or serie.empty :
                continue
            delim = self._conf.tmpdata_column_delim
            serie = serie.dropna( how='all')
            serie = serie.reset_index()
            self.create_output_directory()
            serie.to_csv( self.datapool_filename( serieid, _abs=True), header=True, \
                na_rep='na', sep=delim, index=True, index_label=['row'], date_format='%Y-%m-%dT%H:%M:%S')

