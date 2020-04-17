from bdmcore.selection.method import boruta
from bdmcore.selection.method import correlation
from bdmcore.selection.method import fval
from sklearn.preprocessing import LabelBinarizer
from bdmcore import d_type
from bdmcore import preprocess
import numpy as np
import logging


def setup(zip_val=None):
    """Feature Selection Setting

    :param zip_val: master id and label
    """
    global master_id
    global label
    master_id = zip_val['mid']
    label = zip_val['label']


def run(df, method="boruta", select_x=None, selector=None,
        rank=2,
        threshold=0.7, alpha=0.05, k=250, corr_k_pass=None, mode="chi2"):
    """Feature Selection pipeline

    :param df: dataframe
    :param method: feature selection method
    :param select_x: features selected
    :param selector: feature selection model
    :param rank: select feature ranking
    :param threshold: select feature threshold
    :param alpha: hyper-parameter [alpha]
    :param k: number of select features
    :param corr_k_pass: correlation threshold
    :param mode: feature selection based on static method
    :return: dataframe of feature selected
    """
    print("Shape of df : ", df.shape)
    col_y = label[0]
    if col_y not in list(df.columns):
        raise Exception("Columns not match")
    else:
        y_label = df[col_y]
        if col_y in select_x:
            select_x.remove(col_y)

    lb = LabelBinarizer()
    y = lb.fit_transform(list(y_label))
    y = y.ravel()
    for target in master_id:
        if target in list(df.columns):
            if target in select_x:
                select_x.remove(target)

    df, variable_types, data_types = d_type.run(df[select_x], method="adm")
    df = preprocess.process_tranform(df, data_types)
    df = preprocess.process_nan(df)
    date_col = list(set(list(data_types[data_types['datatype'] == "Date/Time"]['column'])))
    date_col2 = list(set(list(variable_types[variable_types['variabletype'] == "datetime64[ns]"]['column'])))
    df = df.replace([np.inf, -np.inf, np.NaN], 0)

    for col in list(df.columns):
        typ = str(df[col].dtype)
        if (typ == "object") or (typ == 'datetime64[ns]') or (col in date_col) or (col in date_col2):
            logging.info("del col : ")
            logging.info(col)
            logging.info(typ)
            del df[col]
        else:
            logging.info("check col pass : ")
            logging.info(col)
            logging.info(typ)

    number_val = ["uint8", "int32", "int64", "float64", "bool"]
    # re-check datatype
    for col in list(df.columns):
        typ = str(df[col].dtype)
        if typ not in number_val:
            del df[col]

    df.to_csv("check_extract_b4_corr_aftercut")
    if method == "boruta":
        if selector:
            df, feature_importance = boruta(df, y, selector=selector, rank=rank)
            return df, feature_importance
        else:
            raise Exception("No selector model")
    elif method == "corr":
        df_new = correlation(df, y, threshold=threshold, alpha=alpha, corr_k_pass=corr_k_pass, mode=mode)
        return df_new
    elif method == "fval":
        df_new = fval(df, y, alpha=alpha, k=k)
        return df_new


if __name__ == "__main__":
    pass
