import pandas as pd
import numpy as np
from bdmcore.dfs import transform_primitive as tp
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import logging
from bdmcore import preprocess

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def fillnan(df,
            replace_numeric=0,
            replace_bool="MODE",
            replace_date="NAN",
            replace_object="NAN"):
    """Fill nan to dataframe"""
    for col in df:
        nunique = df[df[col].notnull()][col].value_counts().count()
        if nunique == 1:
            del df[col]
        else:
            if df[col].isnull().sum() > 0:
                if nunique == 2:
                    variable_type = 'bool'
                else:
                    variable_type = str(df[col].dtypes)
                if (variable_type == "int64") or (variable_type == "float64"):
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    if replace_numeric == 0:
                        df[col] = df[col].fillna(replace_numeric)
                    elif replace_numeric == "MODE":
                        df[col] = df[col].fillna(df[col].value_counts().idxmax())
                elif variable_type == "bool":
                    if replace_bool == "NAN":
                        df[col] = df[col].fillna(replace_bool)
                    elif replace_bool == "MODE":
                        df[col] = df[col].fillna(df[col].value_counts().idxmax())
                elif variable_type == "datetime64[ns]":
                    if replace_date == "NAN":
                        df[col] = df[col].fillna(replace_date)
                    elif replace_date == "MODE":
                        df[col] = df[col].fillna(df[col].value_counts().idxmax())
                else:
                    if replace_object == "NAN":
                        df[col] = df[col].fillna(replace_object)
                    elif replace_object == "MODE":
                        df[col] = df[col].fillna(df[col].value_counts().idxmax())
    return df


def transform_datetime(df, col_datetime, fn_tf_date=None):
    """Extract DateTime data"""
    if fn_tf_date is None:
        fn_tf_date = [tp.Day(), tp.Month(), tp.Weekday(), tp.Weekend(), tp.Year(), tp.Week()]
    for func in fn_tf_date:
        df_gen = df[col_datetime].agg(func.get_function())
        for col in list(df_gen.columns):
            name = col + '.%s' % func.name
            df[name] = df_gen[col]
    for col in col_datetime:
        df[col] = pd.to_datetime(df[col])
    return df


def tranform_boolean(df, col_boolean):
    """Extract Boolean data"""
    for col in col_boolean:
        if df[col].dtype != bool:
            set_col = list(set(df[col].dropna()))
            set_t = max(set_col)
            set_f = min(set_col)
            list_bool = [-1, 1, 0, -2, 2, "1", "-1", "0", "2", "-2", "Y", "N", "y", "n", "yes", "no"]
            finish_bool = [True, False, "True", "False"]
            if (set_col[0] in finish_bool) and (set_col[1] in finish_bool):
                pass
            elif (set_col[0] in list_bool) and (set_col[1] in list_bool):
                df[col] = np.where(df[col] == set_t, True,
                                   (np.where(df[col] == set_f, False), None))
                df[col] = df[col].astype(bool)
            else:
                del df[col]
    return df


def pre_selection(df, data_types,
                  numeric_encoder=False,
                  skew_encoder=False,
                  bool_encoder=False,
                  drop_datetime=False,
                  cat_encoder="label"):
    """Select feature on Preprocessing based"""
    logging.info("len dataframe : {}".format(len(df)))
    if numeric_encoder:
        logging.info("Processing: Numeric Scaling")
        if data_types['Numeric']:
            min_max = MinMaxScaler()
            df[data_types['Numeric']] = min_max.fit_transform(df[data_types['Numeric']])

    if skew_encoder:
        logging.info("Processing: Skew Scaling")
        if data_types['Skew']:
            df[data_types['Skew']] = scale(df[data_types['Skew']])

    if cat_encoder == "label":
        enc = LabelEncoder()
    elif cat_encoder == "onehot":
        enc = OneHotEncoder(sparse=False)
    else:
        raise Exception("No key %s" % cat_encoder)

    logging.info("Processing: Boolean")
    lb = LabelBinarizer()
    if bool_encoder:
        for column in data_types['Boolean']:
            df[column] = lb.fit_transform(df[column].tolist())

    if cat_encoder:
        for column in list(data_types['Categorical']):
            df_count = (df[column].value_counts())
            df_count = (df_count.to_frame(name=None)).reset_index()
            mean_count = (df_count[column].mean())
            sd_count = (df_count[column].std())
            lst_cut = list(df_count[df_count[column] < (mean_count + (3 * sd_count))]["index"])  #
            df[df[column].isin(lst_cut)][column] = np.NaN

            if cat_encoder == "label":
                df[column] = enc.fit_transform(df[column].tolist())
            elif cat_encoder == "onehot":
                df_temp = pd.get_dummies(df[column])
                temp_col = []
                for col in list(df_temp.columns):
                    temp_col.append('dummy_' + str(column) + '_' + str(col))
                df_temp.columns = temp_col
                del df[column]
                df = pd.concat([df, df_temp], axis=1)
                for check in list(df.isnull().sum()):
                    if check != 0:
                        df_temp = preprocess.process_nan(df)
                        df = df_temp.replace([np.inf, -np.inf, np.NaN], 0)

    logging.info("Processing: Date/Time")
    if drop_datetime:
        df = df.drop(data_types['Date/Time'], axis=1)

    logging.info("Processing: Writing")
    logging.info("Number of features : {}".format(len(list(df.columns))))
    return df


def delete_similar_column(df, na_mode=None, drop_val_col=False, drop_duplicate=False, drop_duplicate_col=False,
                          drop_same_between=False):
    """Drop Nan & Similar Column value"""
    logging.info('Before resize {}'.format(df.shape))
    origin_col = list(df.columns)
    if na_mode == 'all':
        df = df.dropna(thresh=0.9 * df.shape[1], axis=0)
        df = df.dropna(thresh=0.9 * len(df), axis=1)
    elif na_mode == 'column':
        df = df.dropna(thresh=0.9 * len(df), axis=1)
    elif na_mode == 'row':
        df = df.dropna(thresh=0.90 * df.shape[1], axis=0)
    elif None:
        pass

    # Drop same value in column
    if drop_val_col:
        logging.info("drop_val_col : ")
        logging.info(str(df.std()[(df.std() == 0)].index))
        df = df.drop(df.std()[(df.std() == 0)].index, axis=1)

    # Drop duplicate between row
    if drop_duplicate:
        try:
            df = df.drop_duplicates()
        except:
            pass

    # Drop duplicate between col
    if drop_duplicate_col:
        try:
            df = df.iloc[:, ~df.columns.duplicated()]
        except:
            pass

    # Drop same value in row
    if drop_same_between:
        try:
            df_temp = df
            col_select = list(df_temp.T.drop_duplicates().T.columns)
            col_select_df = list(set(col_select + origin_col))
            df = df[col_select_df]
        except:
            df_temp = df.sample(n=500)
            col_select = list(df_temp.T.drop_duplicates().T.columns)
            col_select_df = list(set(col_select + origin_col))
            df = df[col_select_df]
    logging.info('After resize {}'.format(df.shape))
    return df
