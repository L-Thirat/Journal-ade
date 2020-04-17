from bdmcore.clean_data.clean import transform_datetime, tranform_boolean
from bdmcore.clean_data.clean import delete_similar_column
from bdmcore.clean_data.clean import pre_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import logging


def process_nan(df):
    """Deal with Nan pipeline

    :param df: dataframe
    :return: dataframe preprocessed
    """
    df = delete_similar_column(df, na_mode="column", drop_val_col=False, drop_duplicate=False, drop_same_between=False)
    return df


def process_tranform(df, data_types, p_date=True, p_bool=True):
    """Data tranform pipeline

    :param df: dataframe
    :param data_types: data types
    :param p_date: True [to transform datetime data], False [if not]
    :param p_bool: True [to transform boolean data], False [if not]
    :return: dataframe preprocessed
    """
    col_datetime = list(data_types[data_types["datatype"] == 'Date/Time']["column"])
    col_boolean = list(data_types[data_types["datatype"] == 'Boolean']["column"])
    if col_datetime and p_date:
        df = transform_datetime(df, col_datetime)
    if col_boolean and p_bool:
        df = tranform_boolean(df, col_boolean)
    return df


def process_datatype(df, data_types):
    """Pre-process data type pipeline

    :param df: dataframe
    :param data_types: data types
    :return: dataframe preprocessed
    """
    df = pre_selection(df, data_types)
    return df


def preprocess_encoder(df, cat_encoder="label"):
    """Pre-process using encoder pipeline

    :param df: dataframe
    :param cat_encoder: categorical encoder
    :return: dataframe preprocessed
    """
    if cat_encoder == "label":
        enc = LabelEncoder()
    elif cat_encoder == "onehot":
        enc = OneHotEncoder(sparse=False)
    else:
        raise Exception("No encoder: %s" % cat_encoder)
    for column in list(df.columns):
        if str(df[column].dtype) == "object":
            if cat_encoder == "label":

                df[column] = enc.fit_transform(df[column].tolist())
            elif cat_encoder == "onehot":
                logging.info("onehot_%s" % column)
                df_temp = pd.get_dummies(df[column])
                temp_col = []
                for col in list(df_temp.columns):
                    temp_col.append(str(column) + '_' + str(col))
                df_temp.columns = temp_col
                del df[column]
                df = pd.concat([df, df_temp], axis=1)
    return df
