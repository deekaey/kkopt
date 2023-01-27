
from kkplot.kkutils.log import *
import os
import argparse

class kkopt_configuration( object) :

    def  __init__( self) :

        self.basedirvars = dict( table='KKPLOT_TABLESDIR', outputs='KKPLOT_OUTPUTSDIR', tmp='KKPLOT_TMPDIR',
            measurements='KKPLOT_MEASUREMENTSDIR', providers='KKPLOT_PROVIDERSDIR')

        parser = argparse.ArgumentParser()

        parser.add_argument( '--outputs-dir', default=None,
            help='directory where output files are written (default=".", overwrites environment variable `KKPLOT_OUTPUTSDIR\')')
        parser.add_argument( '--tmp-dir', default=None,
            help='directory where output temporary files are written (default=".", overwrites environment variable `KKPLOT_TMPDIR\')')

        parser.add_argument( '--data-dir', default=None,
            help='base directory for data sources (e.g., model outputs), (default=".", overwrites environment variable `KKPLOT_DATADIR\')')
        parser.add_argument( '--measurements-dir', default=None,
            help='base directory for measurement data (e.g., observations of nitrate leaching rates), (default=".", overwrites environment variable `KKPLOT_MEASUREMENTSDIR\')')
        parser.add_argument( '--providers-dir', default=None,
            help='base directory for data providers (e.g., measurements parser), (default=".", overwrites environment variable `KKPLOT_PROVIDERSDIR\')')

        parser.add_argument( '-D','--env', action='append', default=None,
            help='define additional environment variables, can be given multiple times (e.g., -DPROJECTS_DIR=/home/projects)')

        parser.add_argument( '--envf', default=None,
            help='define additional environment variables given in text file')

        parser.add_argument( '--debug', action='store_true', default=False,
            help='switch on debug mode')

        parser.add_argument( '-V', '--version', action='store_true', default=False,
            help='show version')

        parser.add_argument( 'optfile', nargs='?', default='-',
            help='YAML figure description file (default="-" (stdin))')

        self.args = parser.parse_args()

        kklog.set_debug( self.args.debug)
        kklog.set_color( self.args.debug)

        self.set_environment()
        self.set_basedirs()

    @property
    def  showversion( self) :
        return  self.args.version


    def  optfile( self) :
        return  self.args.optfile


    def  outputs_dir( self) :
        return  self.base_dir_for( 'outputs')
    def  tmp_dir( self) :
        return  self.base_dir_for( 'tmp')
    def  data_dir( self) :
        return  self.base_dir_for( 'table')
    def  measurements_dir( self) :
        return  self.base_dir_for( 'measurements')
    def  providers_dir( self) :
        return  self.base_dir_for( 'providers')
    def  base_dir_for( self, _kind) :
        if _kind in self.basedirvars :
            return os.environ.get( self.basedirvars[_kind])
        else :
            raise RuntimeError( 'unknown data kind  [%s]' % ( _kind))

    def  option( self, _key, _delim=None) :
        if _key in self.gopts.keys() :
            gopt = self.gopts[_key].strip()
            if _delim and _delim in gopt :
                return  gopt.split( _delim)
            if _delim :
                return [ gopt]
            return  gopt
        return  ''


    def  __str__( self) :
        return  'kkopt configuration: %s\n' % ( self.args)

    
    def set_basedirs( self) :
        ## "table"
        tablesdir = self.args.data_dir
        if tablesdir is None :
            tablesdir = os.environ.get( 'KKPLOT_DATADIR')
        if tablesdir is None :
            tablesdir = os.environ.get( 'KKPLOT_TABLESDIR')
        if tablesdir is None :
            tablesdir = os.environ.get( 'PLOTTER_SOURCE_PATH')
        if tablesdir is None :
            tablesdir = '.'
        tablesdir = self.normalize_dir( tablesdir)
        os.environ['KKPLOT_DATADIR'] = tablesdir
        os.environ['KKPLOT_TABLESDIR'] = tablesdir

        ## "measurements"
        measurementsdir = self.args.measurements_dir
        if measurementsdir is None :
            measurementsdir = os.environ.get( 'KKPLOT_MEASUREMENTSDIR')
        if measurementsdir is None :
            measurementsdir = os.environ.get( 'PLOTTER_MEASUREMENTS_PATH')
        if measurementsdir is None :
            measurementsdir = '.'
        measurementsdir = self.normalize_dir( measurementsdir)
        os.environ['KKPLOT_MEASUREMENTSDIR'] = measurementsdir

        ## "providers"
        providersdir = self.args.providers_dir
        if providersdir is None :
            providersdir = os.environ.get( 'KKPLOT_PROVIDERSDIR')
        if providersdir is None :
            providersdir = os.environ.get( 'PLOTTER_HOME')
        if providersdir is None :
            providersdir = '.'
        providersdir = self.normalize_dir( providersdir)
        os.environ['KKPLOT_PROVIDERSDIR'] = providersdir

        ## "outputs"
        outputsdir = self.args.outputs_dir
        if outputsdir is None :
            outputsdir = os.environ.get( 'KKPLOT_OUTPUTSDIR')
        if outputsdir is None :
            outputsdir = '.'
        outputsdir = self.normalize_dir( outputsdir)
        os.environ['KKPLOT_OUTPUTSDIR'] = outputsdir

        ## "tmp"
        tmpdir = self.args.tmp_dir
        if tmpdir is None :
            tmpdir = os.environ.get( 'KKPLOT_TMPDIR')
        if tmpdir is None :
            tmpdir = '.'
        tmpdir = self.normalize_dir( tmpdir)
        os.environ['KKPLOT_TMPDIR'] = tmpdir

    def normalize_dir( self, _basedir) :
        return _basedir.replace( '\\', '/').strip()

    def set_environment( self) :
        if self.args.envf is not None :
            if ( exists( self.args.envf)) :
                load_dotenv( self.args.envf, override=True)
        if self.args.env is None :
            return
        for envvar in self.args.env :
            value = ''
            name_and_value = envvar.split( '=', 1)
            if len( name_and_value) == 1 : ## 'name'
                pass
            else : ## 'name=value'
                value = name_and_value[1].strip()
            name = name_and_value[0].strip()
            if name == '' :
                kklog_fatal( 'empty environment variable name')
            os.environ[name] = value
