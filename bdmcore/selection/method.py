import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import Binarizer, scale, MinMaxScaler
import logging


def boruta(df, y, selector, rank):
    """Feature selection based on Boruta

    :param df: dataframe
    :param y: label
    :param selector: feature selection model
    :param rank: select feature ranking
    :return: dataframe of feature selected
    """
    X = df.values
    column_df = list(df.columns)

    selector.fit(X, y)

    rank_list = selector.ranking_.tolist()
    feature_importance = pd.DataFrame(np.array([column_df, rank_list]).T,
                                      columns=['column_name', 'rank'])
    feature_importance['rank'] = feature_importance['rank'].astype(int)
    select_cols = []
    if type(rank) == int:
        select_cols = list(feature_importance[feature_importance['rank'] <= rank]['column_name'])
    elif rank == "grade":
        rank_pass = (feature_importance['rank'].mean() - (feature_importance['rank'].std(ddof=0)))
        if rank_pass > 0:
            select_cols = list(feature_importance[feature_importance['rank'] <= rank_pass]['column_name'])
        else:
            raise Exception("Can not ranking")
    elif rank == "mean":
        rank_pass = feature_importance['rank'].mean()
        if rank_pass > 0:
            select_cols = list(feature_importance[feature_importance['rank'] <= rank_pass]['column_name'])
    elif rank == "all":
        select_cols = list(feature_importance['column_name'])
    return df[select_cols], feature_importance


def correlation(df, y, threshold, alpha, corr_k_pass, mode):
    """Feature selection based on correlation between features

    :param df: dataframe
    :param y: label
    :param threshold: select feature threshold
    :param alpha: hyper-parameter [alpha]
    :param corr_k_pass: correlation threshold
    :param mode: feature selection based on static method
    :return: dataframe of feature selected
    """
    df_out = df.corr()
    col_pass = []
    del_col = []
    if mode == "chi2":
        filter_slect = chi2
    elif mode == "f":
        filter_slect = f_classif
    else:
        raise Exception("No mode: " % mode)
    if alpha:
        x_bin = MinMaxScaler().fit_transform(scale(df))
        fpval = SelectFpr(filter_slect, alpha=alpha).fit(x_bin, y).scores_
        df_sort_fval = pd.DataFrame({"col": list(df.columns), "fval": list(fpval)})
        df_sort_fval = df_sort_fval.sort_values(by=['fval'], ascending=False)
        ranking_col = list(df_sort_fval['col'])
    else:
        ranking_col = list(df.columns)

    for i, col in enumerate(ranking_col):
        if col not in del_col:
            col_pass.append(col)
            del_col = list(
                set(del_col + (list(df_out[col][(df_out[col] > threshold) | (df_out[col] < -threshold)].index))))
        else:
            del_col = list(
                set(del_col + (list(df_out[col][(df_out[col] > threshold) | (df_out[col] < -threshold)].index))))
    del df_out

    logging.info("Del col : %d" % len(del_col))
    logging.info("Passed col : %d" % len(col_pass))
    if corr_k_pass:
        if len(col_pass) > corr_k_pass:
            col_pass = col_pass[:corr_k_pass]
    return df[col_pass]


def fval(df, y, alpha, k):
    """Feature Selection based on F-Value

    :param df: dataframe
    :param y: label
    :param alpha: hyper-parameter [alpha]
    :param k: number of select features
    :return: dataframe of feature selected
    """
    x_bin = MinMaxScaler().fit_transform(scale(df))
    select_chi2 = SelectFpr(chi2, alpha=alpha).fit(x_bin, y)
    select_f_classif = SelectFpr(f_classif, alpha=alpha).fit(df, y)

    chi2_selected = select_chi2.get_support()
    f_classif_selected = select_f_classif.get_support()
    chi2_selected_features = [f for i, f in enumerate(df.columns) if chi2_selected[i]]
    logging.info('Chi2 selected {} features {}.'.format(chi2_selected.sum(), chi2_selected_features))

    f_classif_selected_features = [f for i, f in enumerate(df.columns) if f_classif_selected[i]]
    logging.info('F_classif selected {} features {}.'.format(f_classif_selected.sum(), f_classif_selected_features))
    selected = chi2_selected & f_classif_selected
    logging.info('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [f for f, s in zip(df.columns, selected) if s]
    logging.info(features)
    return df[features]
