"""
# Reusable cleaning routine
# Author: Jah Chaisangmongkon
# All functions in this file will receive a dataframe and metadata as an argument.
# Functions should first verify whether metadata is the one that they need.
# Then they will determine which columns or rows should be dropped, based on particular logic and optional arguments.
# Function should return new df with dropped columns, names of dropped columns, or list of indices dropped
"""

import logging
import pandas as pd

log = logging.getLogger(__file__)


def check_metadata(mdf, column_names):
    """Check metadata exists

    :param mdf: starting meta dataframe
    :param column_names: column names
    :return: True [if metadata exists], False [if not]
    """
    exist = [column for column in column_names if column not in mdf.columns]
    log.info("Column {} missing from metadata".format(str(exist)))

    if len(exist) == 0:
        return True
    else:
        return False


def gen_linage(column_to_drop, routine_name, table_name):
    lineage = [(routine_name, table_name, column) for column in column_to_drop]
    lineage = pd.DataFrame(lineage, columns=['routine_name', 'table_name', 'column_to_drop'])
    return lineage


def base_cleaning(df, mdf, cardinal_thr=1, percent_missing_thr=80):
    """This function performs basic cleaning that every table should go through.

    :param df: starting dataframe
    :param mdf: starting meta dataframe
    :param cardinal_thr: the maximum cardinality allowed
    :param percent_missing_thr: the maximum percent missing value allowed
    :return: cleaned dataframe and columns dropped
    """
    if not check_metadata(mdf, ['cardinality', 'percent_missing']):
        raise Exception('metadata missed required columns.')
    col_to_drops = []
    col_to_drop = mdf.loc[mdf['cardinality'] <= cardinal_thr]['column_name'].tolist()
    col_to_drops.extend(col_to_drop)  # add more
    df = df.drop(col_to_drop, axis=1)

    log.info("Dropping {} columns due to low cardinality: {}"
             .format(len(col_to_drop), str(col_to_drop)))

    col_to_drop = mdf.loc[mdf['percent_missing'] > percent_missing_thr]['column_name'].tolist()
    col_to_drop = set(col_to_drop).intersection(set(df.columns))
    col_to_drop = list(col_to_drop)  # add more
    col_to_drops.extend(col_to_drop)  # add more
    df = df.drop(col_to_drop, axis=1)

    log.info("Dropping {} columns due to high percentage of missing values: {}"
             .format(len(col_to_drop), str(col_to_drop)))

    return df, col_to_drops


def frequent_value_domination(df, domination_thr=0.80):
    """Find columns which is over-dominated by frequent values.

    :param df: input dataframe
    :param domination_thr: the maximum domination ratio allowed
    :return: output dataframe, list of columns to drop
    """

    log.info("Frequent value domination at threshold {}".format(domination_thr))
    col_to_drop = []
    for column in df.columns:
        counts = df[column].value_counts() / len(df)
        if counts.iloc[0] > domination_thr:
            log.info("Column {0:s} is too dominated by value {1:s} at {2:.2f}%".format(
                column, str(counts.index[0]), counts.iloc[0] * 100))
            col_to_drop.append(column)
    log.info("Removing {} columns in the following list due to frequent value domination: {}".format(len(col_to_drop),
                                                                                                     col_to_drop))
    df = df.drop(col_to_drop, axis=1)
    return df, col_to_drop
