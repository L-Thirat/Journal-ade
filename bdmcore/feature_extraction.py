import pandas as pd
from bdmcore.dictionary.problemDefine import entity_discovery, find_main_id, find_forward_backward
from bdmcore.dfs.method import adm
from bdmcore.dfs.method import adm_math
from bdmcore.dfs.method import featuretools


def setup(gb_dic_type, gb_entity_dic=None, gb_relation_table=None, zip_val=None, gb_test_mode=None):
    """Feature Extraction Setting"""
    if gb_test_mode is None:
        gb_test_mode = {}
    global data_files
    global master_id
    global label
    global set_val
    global dic_type
    global ent_dic
    global rt
    global test_mode
    dic_type = gb_dic_type
    test_mode = gb_test_mode

    if gb_relation_table is not None:
        rt = gb_relation_table
    if gb_entity_dic:
        ent_dic = gb_entity_dic
    if zip_val:
        set_val = zip_val
        data_files = zip_val['files']
        master_id = zip_val['mid']
        label = zip_val['label']


def gen_entity_tables(df_dic):
    """Generate entity tables"""
    # Generate entity tables
    entity_dic = entity_discovery(df_dic)
    # Generate relation tables
    relation_table = find_forward_backward(df_dic, entity_dic)
    return entity_dic, relation_table


def pre_generate(df_dic):
    """Pre-Process data to generate entity tables"""
    file_paths = df_dic.keys()
    main_table_id = {}
    for file in file_paths:
        if find_main_id(ent_dic[file]):
            main_table_id[file] = find_main_id(ent_dic[file])
    for file in main_table_id:
        for other_file in main_table_id:
            fake_col = "fake_" + main_table_id[other_file]
            if (other_file != file) and list(set(main_table_id[other_file]) - set(df_dic[file].columns)) \
                    and list(set(main_table_id[other_file]) & set(df_dic[file].columns)):

                col_use = list(main_table_id[file]) + list(main_table_id[other_file])
                df_other_fake = df_dic[other_file][col_use]
                df_other_fake[fake_col] = df_other_fake[main_table_id[other_file]]
                del df_other_fake[main_table_id[other_file]]
                df_dic[file] = pd.merge(df_dic[file], df_other_fake, on=[main_table_id[file]],
                                        how='left').drop_duplicates()
    return df_dic


def merge(df_merge, df_dic, relation_table, target_table, main_tables, mem_merge):
    """Merge Tables"""
    # TODO Auto merge from relation_table, target_table, mem_merge ?
    fw_files = main_tables[1:]
    if fw_files:
        for sub_file in fw_files:
            intersec = list(set(df_merge.columns) & set(df_dic[sub_file].columns))
            df_merge = pd.merge(df_merge, df_dic[sub_file], how='left', on=intersec)
    return df_merge


def run(new_df, df_dic, method, main_tables, deep=4):
    """Feature Extraction pipeline"""
    if method == "adm_extract":
        new_df = adm(new_df, df_dic, dic_type, main_tables, ent_dic, set_val, test_mode)
    elif method == "adm_math":
        new_df = adm_math(new_df, dic_type, test_mode)
    elif method == "featuretool":
        new_df = featuretools(new_df, df_dic, dic_type, deep=deep, main_table=main_tables, entity_dic=ent_dic,
                              relation_table=rt)
    return new_df
