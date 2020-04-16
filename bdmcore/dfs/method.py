from bdmcore.dfs.extract_features import feature_groupby
from bdmcore.dfs.extract_features import feature_math
from bdmcore.dictionary.problemDefine import find_main_id
from bdmcore.clean_data.clean import delete_similar_column
import featuretools as ft
import logging


def adm(new_df, df_dic, dic_type, main_tables, entity_dic, set_val, test_mode):
    """Extract feature from relation of data"""
    return feature_groupby(new_df, df_dic, dic_type, main_tables, entity_dic, set_val, test_mode)


def adm_math(new_df, dic_type, test_mode):
    """Extract feature using math function"""
    return feature_math(new_df, dic_type, test_mode)


def featuretools(new_df, df_dic, dic_type, deep, main_table, entity_dic, relation_table):
    """Extract feature using featuretools"""

    ft.EntitySet(id="featuretools")

    relation_list = []
    for file in list(relation_table.columns):
        if "FW" in list(relation_table[file]):
            link_file = (list(relation_table[relation_table[file] == "FW"]["file"]))
            for lf in link_file:
                if find_main_id(entity_dic[file]) in df_dic[lf]:
                    link_col = find_main_id(entity_dic[file])

                relation_list.append((file, link_col, lf, link_col))

    es = {}
    for file in df_dic:
        if find_main_id(entity_dic[file]):
            idx = find_main_id(entity_dic[file])
        else:
            idx = "fake_idx"
            df_dic[file][idx] = range(0, int(df_dic[file].shape[0]))
        es[file] = (df_dic[file], idx)

    for file in main_table:
        print("Filename: ",file)
        feature_matrix_customers, features_defs = ft.dfs(entities=es, relationships=relation_list, target_entity=file,
                                                         max_depth=deep)
        intersec_col = list(
            set(new_df.columns) & set(feature_matrix_customers.reset_index().columns))
        df_write = (feature_matrix_customers.reset_index())

        new_df = new_df.merge(df_write, how='left', on=intersec_col)
        new_df = delete_similar_column(new_df, drop_val_col=True, drop_duplicate_col=True)

        logging.info(new_df.head(2))
        logging.info("----------------------")
        print("shape new df ", new_df.shape)

    print(new_df.head(2))
    return new_df
