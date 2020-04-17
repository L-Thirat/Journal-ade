import pandas as pd
import numpy as np
import math
import ast
from bdmcore.dfs import aggregation_primitives as ap
from bdmcore.clean_data.clean import delete_similar_column
from bdmcore import feature_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import logging

__all__ = [
    'feature_groupby',
    'feature_math'
]
global k
global model
global feat_selector

# Modeling
k = 300
model = 'rf'
if model == 'rf':
    clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
elif model == 'ln':
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
elif model == 'et':
    clf = ExtraTreesClassifier()
elif model == 'xgb':
    clf = XGBClassifier(learning_rate=1.0)
feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2, random_state=1, max_iter=100)


def feature_groupby(new_df, df_dic, gb_dic_type, main_tables, gb_entity_dic, set_val, test_mode,
                    groupby_max=2, start=1,
                    numeric_fn=None,
                    cat_fn=None,
                    bool_fn=None,
                    datetime_fn=None):
    """Extract aggregation features"""
    if datetime_fn is None:
        datetime_fn = [ap.AvgTimeBetween(), ap.Last()]
    if bool_fn is None:
        bool_fn = [ap.PercentTrue()]
    if cat_fn is None:
        cat_fn = [ap.Count(), ap.Mode(), ap.NUnique(), ap.Zero_count(), ap.Last()]
    if numeric_fn is None:
        numeric_fn = [ap.Mean(), ap.Median(), ap.Std(), ap.Mode(), ap.Zero_count(), ap.Min()]
    global master_id
    global dic_type
    global entity_dic
    global label
    label = set_val['label']
    master_id = set_val['mid']
    dic_type = gb_dic_type
    entity_dic = gb_entity_dic

    pass_group = []
    all_feat = 0
    old_no = 0

    columns_groupby = []
    for file in main_tables:
        columns_groupby = list(set(columns_groupby + dic_type[file]['Not_Numeric']))

    for i in range(start, groupby_max + 1):
        original_col = list(new_df.columns)
        total_feat_out = len(columns_groupby)
        logging.info(total_feat_out)
        now_no = 0

        if total_feat_out >= i:
            all_feat += (math.factorial(total_feat_out) / (math.factorial(total_feat_out - i) * math.factorial(i)))
            set_groupby = powerset(columns_groupby, i)
            total = len(set_groupby) + old_no

            logging.info("Set Groupby: ", set_groupby)
            for ls_groupby in set_groupby:
                if ls_groupby not in pass_group:
                    pass_group.append(ls_groupby)
                    logging.info("Group by : %s" % str(ls_groupby))
                    now_no = set_groupby.index(ls_groupby) + 1 + old_no
                    logging.info("Run: %d,%d|%d,%d" % (i, now_no, total, all_feat))
                    for other_file in df_dic:
                        logging.info("process file: %s" % other_file)
                        for typ in dic_type[other_file]:
                            if typ == 'Categorical':
                                fn = cat_fn
                            elif typ == 'Numeric':
                                fn = numeric_fn
                            elif typ == 'Boolean':
                                fn = bool_fn
                            elif typ == 'Date/Time':
                                fn = datetime_fn
                            else:
                                raise Exception("No datatype: %s" % typ)
                            if (dic_type[other_file][typ]) and (typ != "Not_Numeric"):
                                column_fn = list(set(dic_type[other_file][typ]) - set(ls_groupby))
                                if other_file not in main_tables:
                                    intersect_col = list(set(df_dic[other_file].columns) & set(new_df.columns))
                                    cols_in_typ = dic_type[other_file][typ]
                                    cols_in_typ = list(set(cols_in_typ) & set(df_dic[other_file].columns))
                                    column_other = list(set(cols_in_typ + intersect_col))
                                    new_df_origin = list(set(intersect_col + ls_groupby))
                                    if len(new_df_origin) != len(list(set(new_df[new_df_origin].columns))):
                                        temp_merge_df = new_df[new_df_origin].iloc[
                                                        :, ~new_df[new_df_origin].columns.duplicated()]
                                    else:
                                        temp_merge_df = new_df[new_df_origin]
                                    temp_merge_df = pd.merge(temp_merge_df, df_dic[other_file][column_other],
                                                             on=intersect_col, how='left')
                                    for col_sec in intersect_col:
                                        if col_sec not in ls_groupby:
                                            temp_merge_df = temp_merge_df.drop(col_sec, axis=1)
                                else:
                                    intersect_col = True
                                    cols_in_typ = dic_type[other_file][typ]
                                    cols_in_typ = list(set(cols_in_typ) & set(df_dic[other_file].columns))
                                    column_other = list(set(cols_in_typ + ls_groupby))
                                    temp_merge_df = new_df[column_other]
                                if intersect_col and (len(list(temp_merge_df.columns)) > len(ls_groupby)) and fn:
                                    new_df = concat_feature(new_df=new_df, merge_df=temp_merge_df,
                                                            ls_groupby=ls_groupby, col_fn=column_fn, fn=fn)
                                    if len(list(new_df.columns)) != len(list(set(new_df.columns))):
                                        logging.info(new_df)
                                        raise Exception("Shape not match")

                                    newfeats_col = list(set(new_df.columns) - set(original_col))
                                    if len(newfeats_col) > k / 8:
                                        if not test_mode:
                                            logging.info("shape before select by corr : %s" % str(len(newfeats_col)))
                                            selected_new_df = feature_selection.run(df=new_df, method='corr',
                                                                                    select_x=newfeats_col,
                                                                                    threshold=0.65,
                                                                                    corr_k_pass=int(k / 5))
                                            logging.info(
                                                "shape after select by corr : %s" % str(selected_new_df.shape))
                                            new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)
                                            if len(list(new_df.columns)) != len(list(set(new_df.columns))):
                                                logging.info(new_df)
                                                raise Exception("Shape not match")
                                            logging.info("shape merge select by corr : %s" % str(new_df.shape))
                                        else:
                                            file = main_tables[0]
                                            file_out = file.replace("test", "train")
                                            file_out = '../feature/group_list_%s_%d.txt' % (file_out, i)
                                            logging.info("Reading : %s" % file_out)
                                            f = open(file_out, 'r')
                                            intersect_col = list(new_df.columns & set(ast.literal_eval(f.read())))
                                            new_df = new_df[intersect_col]
                                            logging.info("selected from train cols list :%s" % str(new_df.columns))
                                            f.close()
        logging.info('# Extract Group %d Finished' % i)
        old_no = now_no
        if not test_mode:
            file = main_tables[0]
            file_out = file.replace("test", "train")
            file_out = '../feature/group_list_%s_%d.txt' % (file_out, i)
            logging.info("Writing : %s" % file_out)
            f = open(file_out, 'w')
            f.write(str(list(new_df.columns)))
            f.close()

    if not test_mode:
        logging.info("Cleaning..")
        logging.info("not concat0 = %s" % str(set(new_df.columns) - (set(new_df.columns) & set(original_col))))
        new_df = delete_similar_column(new_df, drop_val_col=True)  # drop some original_col
        original_col = list(set(original_col) & set(new_df.columns))

        logging.info(original_col)
        newfeats_col = list(set(new_df.columns) - set(original_col))
        logging.info("shape before select by corr : %s" % str(len(newfeats_col)))
        logging.info(new_df.dtypes)
        logging.info("not concat1 = %s" % str(set(new_df.columns) - (set(new_df.columns) & set(original_col))))
        selected_new_df = feature_selection.run(df=new_df, method='corr', select_x=newfeats_col, threshold=0.65)
        logging.info("shape after select by corr : %s" % str(selected_new_df.shape))
        new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)
        logging.info("shape after concat : %s" % str(new_df.shape))
        logging.info("not concat2 = %s" % str(set(new_df.columns) - (set(new_df.columns) & set(original_col))))
        newfeats_col = list(set(new_df.columns) - set(original_col))
        if len(newfeats_col) > (k / 10):
            selected_new_df, _ = feature_selection.run(df=new_df, method="boruta", select_x=newfeats_col,
                                                       selector=feat_selector, rank=int(k / 10))  # int(k/15)
            logging.info("not concat = %s" % str(set(new_df.columns) - (set(new_df.columns) & set(original_col))))
            new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)
            logging.info("shape after boruta : %s" % str(new_df.shape))

        file_out = '../feature/group_list_corr.txt'
        logging.info("Writing : ", file_out)
        f = open(file_out, 'w')
        f.write(str(list(new_df.columns)))
        f.close()
    else:
        file_out = '../feature/group_list_corr.txt'
        logging.info("Reading : %s" % file_out)
        f = open(file_out, 'r')
        intersect_col = list(new_df.columns & set(ast.literal_eval(f.read())))
        new_df = new_df[intersect_col]
        f.close()
        logging.info("selected from train cols list :%s" % str(new_df.columns))
    return new_df


def feature_math(new_df, dic_type, test_mode):
    """Extract Mathematics features"""
    temp_col = list(new_df.columns)
    original_col = temp_col
    logging.info("shape before extract math : %s" % str(new_df.shape))
    col_train = []
    columns_numeric = list(set(dic_type['Numeric']) & set(new_df.columns))
    logging.info("numeric features: ")
    logging.info(columns_numeric)
    if len(columns_numeric) > 1:
        feat_list_pair = powerset(columns_numeric, group_lv=2)
        all_feat = len(feat_list_pair)
        for pair in feat_list_pair:
            now_no = feat_list_pair.index(pair) + 1
            logging.info("%d/%d" % (now_no, all_feat))
            if now_no < all_feat:
                try:
                    new_feature_name = "%s_multi(%s)" % (pair[0], pair[1])
                    if (new_feature_name in col_train) or (col_train == []):
                        new_df[new_feature_name] = new_df[pair[0]] * new_df[pair[1]]
                        new_df[new_feature_name] = new_df[new_feature_name].replace([np.inf, -np.inf], np.nan)
                        if new_df[new_feature_name].isnull().sum() / new_df.shape[0] > 0.1:
                            del new_df[new_feature_name]

                    new_feature_name = "%s_devide(%s)" % (pair[0], pair[1])
                    if (new_feature_name in col_train) or (col_train == []):
                        new_df[new_feature_name] = new_df[pair[0]] * new_df[pair[1]]
                        new_df[new_feature_name] = new_df[new_feature_name].replace([np.inf, -np.inf], np.nan)
                        if new_df[new_feature_name].isnull().sum() / new_df.shape[0] > 0.1:
                            del new_df[new_feature_name]
                except MemoryError:
                    logging.info("Memory Error")
                    raise
                newfeats_col = list(set(new_df.columns) - set(original_col))
                if len(newfeats_col) > k:
                    if not test_mode:
                        logging.info("shape before select by corr : %s" % str(len(newfeats_col)))
                        selected_new_df = feature_selection.run(df=new_df, method='corr', select_x=list(newfeats_col),
                                                                threshold=0.65, corr_k_pass=int(k / 2), mode="f")
                        logging.info("shape after select by corr : %s" % str(selected_new_df.shape))
                        new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)
                        logging.info("shape merge select by corr : %s" % str(new_df.shape))
                    else:
                        file_out = '../feature/math_list.txt'
                        logging.info("Reading : %s" % file_out)
                        f = open(file_out, 'r')
                        intersect_col = list(new_df.columns & set(ast.literal_eval(f.read())))
                        selected_col = list(set(original_col + intersect_col))
                        new_df = new_df[selected_col]
                        logging.info("selected from train cols list :%s" % str(new_df.columns))
                        f.close()

        if not test_mode:
            logging.info("cleaning...")
            newfeats_col = list(set(new_df.columns) - set(original_col))
            selected_new_df = delete_similar_column(new_df[newfeats_col], drop_val_col=True)
            new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)

            newfeats_col = list(set(new_df.columns) - set(original_col))
            logging.info("shape before select by corr : %s" % str(len(newfeats_col)))
            logging.info(new_df.dtypes)
            if len(newfeats_col) > 0:
                selected_new_df = feature_selection.run(df=new_df, method='corr', select_x=list(newfeats_col),
                                                        threshold=0.65, mode="f")
                logging.info("shape after select by corr : %s" % str(selected_new_df.shape))
                new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)
                newfeats_col = list(set(new_df.columns) - set(original_col))
                if len(newfeats_col) > (k / 10):
                    selected_new_df, _ = feature_selection.run(df=new_df, method="boruta", select_x=newfeats_col,
                                                               selector=feat_selector, rank=int(k / 10))
                    new_df = pd.concat([new_df[original_col], selected_new_df], axis=1)

            file_out = '../feature/math_list.txt'
            logging.info("Writing : ", file_out)
            f = open(file_out, 'w')
            f.write(str(list(new_df.columns)))
            f.close()
        else:
            file_out = '../feature/math_list.txt'
            logging.info("Reading : ", file_out)
            f = open(file_out, 'r')
            intersect_col = list(new_df.columns & set(ast.literal_eval(f.read())))
            new_df = new_df[intersect_col]
            f.close()
    return new_df


def powerset(iterable, group_lv):
    """Calculate powerset

    :param iterable: set of data
    :param group_lv: number of grouping
    :return column names
    """
    iterable = sorted(iterable)
    columns = []
    if group_lv == 1:
        for col in iterable:
            columns.append([col])
    elif group_lv == 2:
        for i in range(0, len(iterable) - 1):
            for j in range(i + 1, len(iterable)):
                columns.append([iterable[i], iterable[j]])
    elif group_lv == 3:
        for i in range(0, len(iterable) - 2):
            for j in range(i + 1, len(iterable) - 1):
                for m in range(i + 2, len(iterable)):
                    columns.append([iterable[i], iterable[j], iterable[m]])
    return columns


def concat_feature(new_df, merge_df,
                   ls_groupby, col_fn, fn):
    """Concat dataframe

    :param new_df: main dataframe
    :param merge_df: merge dataframe
    :param ls_groupby: columns which use to merge dataframe
    :param col_fn: columns which use to aggregate
    :param fn: aggregate function
    :return: dataframe
    logging.info("Shape before new dataframe : %s" % str(new_df.shape))
    df_news_feat = create_newfeat(ls_groupby, col_fn, fn, merge_df)
    col_new = list(
        set(list(set(df_news_feat.columns) - (set(df_news_feat.columns) & set(new_df.columns))) + ls_groupby))
    df = pd.merge(new_df, df_news_feat[col_new], on=ls_groupby, how='left')
    logging.info("Shape of new dataframe : %s" % str(df.shape))
    return df


def create_newfeat(ls_groupby, col_fn, fn, merge_df):
    """Create new features

    :param merge_df: merge dataframe
    :param ls_groupby: columns which use to merge dataframe
    :param col_fn: columns which use to aggregate
    :param fn: aggregate function
    :return: dataframe
    """
    fns = []
    col_name = []
    for class_fn in fn:
        fns.append(class_fn.get_function())
    column_fn = list(set(col_fn) & set(merge_df.columns))
    for col in column_fn:
        for class_fn in fn:
            text_col = "%s_%s_%s" % (str(ls_groupby), col, class_fn.name)
            col_name.append(text_col)
    df = merge_df.groupby(ls_groupby, as_index=False).aggregate(fns)
    df.columns = col_name
    df = df.reset_index()
    return df
