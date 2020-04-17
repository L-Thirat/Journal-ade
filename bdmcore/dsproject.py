import configparser
import os
import sys
import metadata_routines

from sqlalchemy import create_engine, MetaData
import pandas as pd
import logging
import logging.config
import json
import datetime
import glob
import boto3

from pandas.io.sql import SQLTable, pandasSQL_builder
from io import StringIO, BytesIO
import gzip
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # <-- ADD THIS LINE
import numpy as np
import inspect
import traceback


class Config:
    """Project setting"""
    def __init__(self, project_name):
        self.project_name = project_name
        self.database_name = project_name
        self.config = self.setting()

    def setting(self):
        '''
        Setting up config file
        :return: config object
        '''
        home_path = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
        s3_path = 's3://' + self.project_name
        user = '<username>'
        password = '<password>'
        redshift_path = 'postgresql://' + user + ':' + password + \
                        '@automl-instance.cbfploinlvqm.us-east-1.redshift.amazonaws.com:5439/' + self.database_name

        config = configparser.ConfigParser()
        config.add_section('PATH')
        config.set('PATH', 'home_path', home_path)
        config.set('PATH', 'model_path', os.path.join(home_path, 'model'))
        config.set('PATH', 'preprocess_path', os.path.join(home_path, 'preprocess'))
        config.set('PATH', 'log_path', os.path.join(home_path, 'log-database'))
        config.set('PATH', 's3_path', s3_path)
        config.set('PATH', 'redshift_path', redshift_path)
        return config

    def get(self, section, param):
        config = self.config.get(section, param)
        return config

    def prediction_setting(self, data_files, master_id, label):
        self.data_files = data_files
        self.master_id = master_id
        self.label = label


class Logger:
    """Logger management"""

    def __init__(self, name, config=None):
        self.name = name
        self.setup_logging(config)
        self.logger = logging.getLogger(name)

    def setup_logging(self, config):
        """Logging setup"""
        if config != None:
            self.log_path = config.get('PATH', 'log_path')
            self.home_path = config.get('PATH', 'home_path')
            logging_config_path = os.path.join(os.path.dirname(os.path.dirname(self.home_path)), 'bdmcore/logging.json')
        else:
            logging_config_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'bdmcore/logging.json')

        if os.path.exists(logging_config_path):
            # load json configuration file
            with open(logging_config_path, 'rt') as f:
                print('open')
                logging_config = json.load(f)

            # adding appropriate log folder to file handler
            log_filename = logging_config['handlers']['info_file_handler']['filename']

            # add time stamp to the filename
            timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
            log_filename = self.name + '_' + log_filename + '_' + timestamp + '.log'

            for handler in logging_config['handlers'].keys():
                if 'filename' in logging_config['handlers'][handler].keys():
                    logging_config['handlers'][handler]['filename'] = os.path.join(self.log_path, log_filename)

            logging.config.dictConfig(logging_config)
        else:
            # create logger with 'spam_application'
            logger = logging.getLogger(self.name)
            logger.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(logger.name + '-logging.log', mode='w')
            fh.setLevel(logging.DEBUG)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            # add the handlers to the logger
            logger.addHandler(ch)
            logger.addHandler(fh)

    def write(self, message=None, log_type=None):
        """Write log"""
        if (log_type == 'info') | (log_type is None):
            self.logger.info(message)
        elif log_type == 'debug':
            self.logger.debug(message)
        elif log_type == 'warning':
            self.logger.warning(message)
        elif log_type == 'error':
            self.logger.error(message)
        elif log_type == 'critical':
            self.logger.critical(message)

    def read(self):
        """Read log"""
        list_of_files = glob.glob(os.path.join(self.log_path, '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file) as f:
            for line in f:
                print(line)


class RedshiftConnector:
    """Redshift Management"""
    # todo shift to logger instead of print
    engine = None

    def __init__(self, config):
        self.s3_path = config.get('PATH', 's3_path')
        self.redshift_path = config.get('PATH', 'redshift_path')
        self.engine = create_engine(self.redshift_path)
        # todo add project_name to connector

    def read(self, table_name, routine_name=None, sub_routine=None, **kwarg):
        if ((sub_routine == None) & (routine_name != None)):
            table_name = routine_name + '/' + table_name
        elif ((sub_routine == None) & (routine_name == None)):
            table_name = table_name
        else:
            table_name = routine_name + '/' + sub_routine + '/' + table_name
        df = pd.read_sql('SELECT * FROM "' + table_name + '" ;', con=self.engine, **kwarg)
        df = df.replace('', np.NaN)
        logging.info('Reading table {} :'.format(table_name))
        return df

    def write(self, data_frame, routine_name, table_name, bucketname=None, if_exists='replace', sub_routine=None,
              **kwarg):
        # todo this function is pretty verbose as it is, please use logger instead of print
        # todo make sure log statement is understandable for outside observer
        # todo bucketname should always be project_name, redshift should know its own project_name
        # todo when table is new, write metadata, but give an option to skip metadata

        self.bucket = bucketname
        if (table_name != 'meta_database') & (sub_routine == None):
            table_name = routine_name + '/' + table_name
        elif (table_name == 'meta_database') & (sub_routine == None):
            table_name = table_name
        else:
            table_name = routine_name + '/' + sub_routine + '/' + table_name
        print(table_name)
        logging.info('Writing table {} :'.format(table_name))

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucketname)
        obj = s3.Object(bucket_name=bucket, key='/')

        con = psycopg2.connect(self.redshift_path)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # <-- ADD THIS LINE
        cur = con.cursor()

        # write DF to string stream
        csv_buffer = StringIO()
        data_frame.to_csv(csv_buffer, index=None, header=None, sep='|')

        # reset stream position
        csv_buffer.seek(0)
        # create binary stream
        gz_buffer = BytesIO()

        # compress string stream using gzip
        with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
            gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))

        # write stream to S3
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
        bucket.put_object(Key='tmp_' + timestamp + '.gz', Body=gz_buffer.getvalue())
        print('saved file ')

        # CREATE THE COPY STATEMENT TO SEND FROM S3 TO THE TABLE IN REDSHIFT
        s3_path_tmp_file = 's3://{0}/{1}'.format(bucketname, 'tmp_' + timestamp + '.gz')

        print('create table')
        table = SQLTable(table_name, pandasSQL_builder(self.engine, schema=None), data_frame, if_exists=if_exists,
                         index=None)

        statements = []
        if table.exists():
            if if_exists == 'fail':
                raise ValueError("Table Exists")
            elif if_exists == 'append':
                statements = []
            elif if_exists == 'replace':
                statements = [""" truncate "{}"; rollback; drop table "{}";""".format(table_name, table_name)]
            else:
                raise ValueError("Bad option for `if_exists`")
        statements.append(table.sql_schema() + ';')

        statement = """
                copy "{0}"
                from '{1}'
                delimiter '{2}'
                region 'us-east-1'
                CREDENTIALS 'aws_access_key_id={3};aws_secret_access_key={4}'
                FORMAT AS CSV NULL AS '@NULL@'
                GZIP
                TRUNCATECOLUMNS
                """.format(table_name, s3_path_tmp_file, '|', 'AKIAIVCDQREXD2TPPRAQ',
                           'SCemMCgkq1rUruSrIDbFdjorHthnvY6E4j8/UEfg')
        statements.append(statement)

        try:
            logging.info('excucte statement')
            for stmt in statements:
                print(stmt)
                cur.execute(stmt)
                # con.commit()
            logging.info('finish execute')

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            con.rollback()
            raise

        s3.Object(bucketname, 'tmp_' + timestamp + '.gz').delete()
        logging.info('FILLING THE TABLE IN REDSHIFT')
        logging.info('\n--------------- write already -----------------')

    def delete(self, table_name, routine_name=None, sub_routine=None):
        if (sub_routine is None) & (routine_name is not None):
            table_name = routine_name + '/' + table_name
        elif (sub_routine is None) & (routine_name is not None):
            table_name = table_name
        else:
            table_name = routine_name + '/' + sub_routine + '/' + table_name

        redshift_mysql_statement = """ DROP TABLE "%s" """ % (table_name)
        meta_mysql_statement = """ DELETE FROM meta_database WHERE tablename='%s' """ % (table_name)
        con = psycopg2.connect(self.redshift_path)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # <-- ADD THIS LINE
        cur = con.cursor()
        cur.execute(redshift_mysql_statement)
        cur.execute(meta_mysql_statement)


class Metadata:
    """Meta data Management"""
    table_name_meta = None

    def __init__(self, config, redshift_connector):
        self.config = config
        self.redshift_connector = redshift_connector
        self.setup_meta()

    def setup_meta(self):
        mysql_statement = """                         
                           CREATE TABLE IF NOT EXISTS meta_database( 
                                tablename varchar(255), 
                                "column" varchar(255),
                                type varchar(255)
                          );
                          """
        con = psycopg2.connect(self.config.get('PATH', 'redshift_path'))
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # <-- ADD THIS LINE
        cur = con.cursor()
        cur.execute(mysql_statement)

    def list_folder(self):
        """List names of all data folders in the project.

        :return: folder_list in dataframe format
        """

    def list_table(self, input_path=None, exclude_keywords=None):
        """List names of all data tables available in a project.
        metdata.list_table() will list all tables in the project.
        To list a subset of tables use input_path.
        To exclude some tables from the list, use keywords.
        To get all raw unmodified data set exclude_keywords = ['/', 'meta_database']

        :param input_path: input_path is table prefix for example input_path = 'routine/subroutine/'
                    will list all tables with routine/subroutine prefix.
        :param exclude_keywords: filter out tables that contain a keyword within this list.

        :return: table_list in dataframe format
        """

        statement = "select distinct(tablename) from pg_table_def where schemaname = 'public' and tablename != 'meta_" \
                    "database' ;"
        table_list = pd.read_sql(statement, self.redshift_connector.engine)
        table_name_list = table_list['tablename'].tolist()
        if input_path:
            table_name_list = [table_name for table_name in table_name_list if table_name.startswith(input_path)]
        if exclude_keywords:
            for kw in exclude_keywords:
                table_name_list = [table_name for table_name in table_name_list if kw not in table_name]
        table_list = table_list.loc[table_list['tablename'].isin(table_name_list)]
        return table_list

    def read(self, routine_name=None):
        if routine_name is None:
            statement = "select * from meta_database"
        else:
            statement = "select * from meta_database where tablename LIKE '" + routine_name + "%%' ;"
        meta_data = pd.read_sql(statement, self.redshift_connector.engine)
        return meta_data

    def write(self, df, routine_name, table_name, sub_routine=None):

        if sub_routine is None:
            table_name = routine_name + '/' + table_name
        else:
            table_name = routine_name + '/' + sub_routine + '/' + table_name

        statement = "select * from meta_database;"
        meta_data = pd.read_sql(statement, self.redshift_connector.engine)
        reds_columns = ['tablename', 'column', 'type']
        left_a = meta_data.set_index(reds_columns)
        right_a = df.set_index(reds_columns)
        res = left_a.reindex(columns=left_a.columns.union(right_a.columns))
        res.update(right_a)
        new_meta_data = res.reset_index().sort_values(by=['tablename'])
        self.redshift_connector.write(data_frame=new_meta_data, table_name='meta_database', routine_name='-',
                                      bucketname=self.redshift_connector.bucket)

    def construct(self, routine_name, table_name, sub_routine=None):
        if sub_routine is None:
            table_name = routine_name + '/' + table_name
        else:
            table_name = routine_name + '/' + sub_routine + '/' + table_name

        statement = "select pg_table_def.tablename, pg_table_def.column, pg_table_def.type \
                     from pg_table_def where pg_table_def.schemaname = 'public' and pg_table_def.tablename = '" + table_name + "' ;"
        raw_meta_data = pd.read_sql(statement, self.redshift_connector.engine)

        if raw_meta_data.empty:
            print('This table does not exit')
            meta_data_update = raw_meta_data
        else:
            statement = "select * from meta_database where tablename = '" + table_name + "' ; "
            meta_data = pd.read_sql(statement, self.redshift_connector.engine)
            if meta_data.empty:
                # add meta_database if not available
                meta_data_update = raw_meta_data
                meta_data_update.to_sql("meta_database", self.redshift_connector.engine, if_exists='append',
                                        index=False)
            else:
                # update meta_database if available
                meta_data_update = meta_data.merge(raw_meta_data, how='right', on=['tablename', 'column', 'type'])
                statement = "select * from meta_database where tablename != '" + table_name + "' ; "
                meta_data = pd.read_sql(statement, self.redshift_connector.engine)
                meta_data = meta_data.append(meta_data_update)
                self.redshift_connector.write(data_frame=meta_data, table_name='meta_database', routine_name='-',
                                              bucketname=self.redshift_connector.bucket)

        return meta_data_update

    @staticmethod
    def list_routines():
        all_functions = inspect.getmembers(metadata_routines, inspect.isfunction)
        routines = []
        for function_name, function in all_functions:
            routines.append(function_name)
        return routines

    def exist(self, table_name, field_name=None):
        routines = []
        if field_name is None:
            statement = " select * from meta_database where tablename like '%%" + table_name + "' ;"
        else:
            statement = " select * from meta_database where tablename like '%%" + table_name + \
                        "' and meta_database.column = '" + field_name + "';"
        meta_data = pd.read_sql(statement, self.redshift_connector.engine)
        for table_name_meta in meta_data.tablename.unique().tolist():
            routine = table_name_meta.split('/')[0]
            routines.append(routine)
        return routines


class S3Connector:
    """To connect with s3"""

    def __init__(self, bucket_name):
        self.bucket = bucket_name

    def save_csv_file(self, data_frame, routine_name, file_name, sub_routine=None, **kwarg):
        """Write csv"""
        if sub_routine is None:
            key = 'output/' + routine_name + '/' + file_name + '.csv'
        else:
            key = 'output/' + routine_name + '/' + sub_routine + '/' + file_name + '.csv'

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)

        # write dataframe to string stream
        csv_buffer = StringIO()
        data_frame.to_csv(csv_buffer, **kwarg)
        csv_buffer.seek(0)  # reset stream position
        bucket.put_object(Key=key, Body=csv_buffer.getvalue())

    def read_csv_file(self, folder_name, file_name, sub_routine=None, **kwarg):
        """Read csv"""
        if sub_routine is None:
            key = folder_name + '/' + file_name + '.csv'
        else:
            key = folder_name + '/' + sub_routine + '/' + file_name + '.csv'

        s3 = boto3.resource('s3')
        obj = s3.Object(self.bucket, key)
        df = pd.read_csv(BytesIO(obj.get()['Body'].read()), **kwarg)
        return df
