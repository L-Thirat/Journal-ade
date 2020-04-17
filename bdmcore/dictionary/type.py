from bdmcore.dictionary.datadict import PrimaryType
import pandas as pd
import logging


def declare_data_type(df, drop_text):
    """Confirm data type

    :param df: datafeame
    :param drop_text: column name of text data
    :return dataframe preprocessed, variable types
    """
    data_types = {'Numeric': [],
                  'Boolean': [],
                  'ID': [],
                  'Categorical': [],
                  'Date/Time': [],
                  'Text': []}
    df_datatype = pd.DataFrame(columns=['column', 'datatype'])
    for col in list(df.columns):
        find_dt = PrimaryType().characterize(series=df[col])
        if find_dt != "Unknown":
            df_append = pd.DataFrame({'column': [col], 'datatype': [find_dt]})
            df_datatype = df_datatype.append(df_append)
            data_types[find_dt].append(col)
        else:
            logging.info("Unknown")
            logging.info(df[col].head(2))
            del df[col]
    if drop_text:
        if data_types['Text']:
            for col in data_types['Text']:
                del df[col]
                print("Delete text column : ", col)
    return df, df_datatype


def declare_variable_type(df):
    """Confirm variable type

    :param df: datafeame
    :return dataframe preprocessed, variable types
    """
    variable_types = {'uint8': [],
                      'int32': [],
                      'int64': [],
                      'float64': [],
                      'object': [],
                      'bool': [],
                      'datetime64[ns]': []
                      }
    df_variable = pd.DataFrame(columns=['column', 'variabletype'])
    for col in list(df.columns):
        if (str(df[col].dtype) == 'int32') or (str(df[col].dtype) == 'uint8'):
            df[col] = df[col].astype('int64')
        variable_types[str(df[col].dtype)].append(col)
        df = check_true_object_float_int(df, variable_types)
        df = check_true_float_int(df, variable_types)
        df_append = pd.DataFrame({'column': [col], 'variabletype': [str(df[col].dtype)]})
        df_variable = df_variable.append(df_append)
    return df, df_variable


def check_true_float_int(df, variable_types):
    """Confirm between float or int

    :param df: datafeame
    :param variable_types: variable types
    :return dataframe
    """
    for col in variable_types['float64']:
        try:
            df_new = (df[col].astype(int)).copy()
            df_new_compare = df[col] - df_new
            if len(list(set(df_new_compare))) == 1:
                df[col] = df[col].astype(int)
        except:
            pass
    return df


def check_true_object_float_int(df, variable_types):
    """Confirm between float or object

    :param df: datafeame
    :param variable_types: variable types
    :return dataframe
    """
    for col in variable_types['object']:
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    return df
