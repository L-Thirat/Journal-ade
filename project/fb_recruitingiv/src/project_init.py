import sys

sys.path.append("/home/ubuntu/core")
from bdmcore.dsproject import Config
from bdmcore.dsproject import Metadata
from bdmcore.dsproject import RedshiftConnector
from bdmcore.dsproject import Logger

from bdmcore import d_type
from bdmcore import preprocess
from bdmcore import feature_extraction
from bdmcore import feature_selection

from bdmcore.dictionary.problemDefine import find_main_table

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Local-Sv setting
sv_test = False
# False = test mode, True = train mode
test_mode = False

# Project setting
ProjectName = "fb_recruitingiv"
dtype_routine = "adm"
feature_extraction_routine = ["adm_extract", "adm_math"]
feature_selection_routine = "boruta"
master_id = ["bidder_id"]
label = ["outcome"]
data_files = ['h_train_csv', 'h_micro_bids_csv']
set_val = {"files": data_files, "mid": master_id, "label": label}
feature_selection.setup(set_val)

# Config
bucket_name = "fb-RecruitingIV"
conf = Config(ProjectName)
redshif = RedshiftConnector(config=conf)
metadata = Metadata(conf, redshif)
logger_info = Logger(ProjectName, conf)
reds_columns = ['tablename', 'column', 'type']

# Src Run
# Preprocess
df_dic = {}
df_meta = None
old_construct = None
logger_info.write("-->PREPROCESS")
if dtype_routine == "adm":
    for origin_file in data_files:
        logger_info.write("Reading Data:(%s)..." % origin_file)
        file = "pre_" + origin_file
        df_dic[file] = redshif.read(origin_file)
        df_dic[file] = preprocess.process_nan(df_dic[file])
        df_dic[file], variable_types, data_types = d_type.run(df_dic[file], method=dtype_routine)
        df_dic[file] = preprocess.process_tranform(df_dic[file], data_types)
        df_dic[file], variable_types, data_types = d_type.run(df_dic[file], method=dtype_routine)
        df_meta = pd.merge(variable_types, data_types, on=["column"])
        df_meta["tablename"] = "%s/%s" % (dtype_routine, file)
        logger_info.write("Writing Preprocess Data:(%s)..." % file)
        logger_info.write(df_dic[file].head())
        if sv_test:
            redshif.write(data_frame=df_dic[file], routine_name=dtype_routine, table_name=file,
                          bucketname=bucket_name)
        logger_info.write("Reading meta data:(%s)... " % file)
        if not test_mode:
            old_construct = metadata.construct(routine_name=dtype_routine, table_name=file)
            old_construct = pd.merge(old_construct[reds_columns], df_meta, how="left", on=["tablename", "column"])
            logger_info.write("Writing meta data:(%s)... " % file)
            metadata.write(old_construct, routine_name=dtype_routine, table_name=file)

if sv_test:
    del df_dic
    df_dic = {}
if df_meta:
    del df_meta
if old_construct:
    del old_construct

# FEATURE EXTRACTION
logger_info.write("-->FEATURE EXTRACTION")

# Entity & Relation Table
for file in data_files:
    file = "pre_" + file
    logger_info.write("Reading Data:(%s)..." % file)
    if sv_test:
        if test_mode:
            train_file = reds_columns[file]
        else:
            train_file = file
        df_dic[file] = redshif.read(table_name=train_file, routine_name=dtype_routine)
    logger_info.write(df_dic[file].dtypes)
entity_dic, relation_table = feature_extraction.gen_entity_tables(df_dic)
logger_info.write("Entity table >")
logger_info.write(entity_dic)
logger_info.write("Relation table >")
logger_info.write(relation_table)

# check update
main_tables = find_main_table(entity_dic, master_id)
first_new_df_file = '%s' % main_tables[0]
new_df = feature_extraction.merge(df_merge=df_dic[first_new_df_file], df_dic=df_dic,
                                  relation_table=relation_table, target_table=first_new_df_file,
                                  main_tables=main_tables,
                                  mem_merge=list(data_files[0]))
if sv_test:
    redshif.write(data_frame=new_df, routine_name="adm", table_name="merge", bucketname=bucket_name)  # todo for REAL !
    original_df = None
else:
    original_df = new_df.copy()

# Load DataType & VariableType
dic_type = {}
temp_type = {}
all_type = ['Numeric', 'Categorical', 'Boolean', 'Date/Time', 'ID']
for file in df_dic:
    logger_info.write("Load dataframe-type :(%s)..." % file)
    temp_type[file] = metadata.construct(routine_name=dtype_routine, table_name=file)
    logger_info.write(temp_type[file])
    dic_type[file] = {}
    for typ in all_type:
        col_type = set(list(temp_type[file][temp_type[file]['datatype'] == typ]['column']))
        dic_type[file][typ] = list(col_type - set(label) - set(master_id))
    dic_type[file]['Categorical'] = dic_type[file]['Categorical'] + dic_type[file]['ID']
    dic_type[file]['Not_Numeric'] = dic_type[file]['Categorical'] + dic_type[file]['Boolean']
    del dic_type[file]['ID']
    logger_info.write(dic_type)

# Extract Feature
feature_extraction.setup(dic_type, entity_dic, relation_table, set_val)
for i, method in enumerate(feature_extraction_routine):
    if i > 0:
        file = "feat_%s" % feature_extraction_routine[i - 1]
        logger_info.write("Reading Data:(%s)..." % file)
        dic_type = {}
        logger_info.write("Load dataframe-type :(%s)..." % file)
        temp_type = metadata.construct(routine_name=feature_extraction_routine[i - 1], table_name=file)
        logger_info.write(temp_type)
        for typ in all_type:
            col_type = set(list(temp_type[temp_type['datatype'] == typ]['column']))
            dic_type[typ] = list(col_type - set(label) - set(master_id))
        dic_type['Not_Numeric'] = dic_type['Categorical'] + dic_type['Boolean']
        del dic_type['ID']
        logger_info.write(dic_type)
        feature_extraction.setup(dic_type)
    new_df = feature_extraction.run(new_df, df_dic, method, main_tables)
    logger_info.write("New features_%s:(%s)" % (method, file))
    logger_info.write(new_df.head())
    new_df_file = "feat_%s" % (feature_extraction_routine[i])
    logger_info.write("Update features...")
    new_df = preprocess.process_nan(new_df)
    new_df, variable_types, data_types = d_type.run(new_df, method=dtype_routine)
    new_df = preprocess.process_tranform(new_df, data_types)
    new_df, variable_types, data_types = d_type.run(new_df, method=dtype_routine)
    df_meta = pd.merge(variable_types, data_types, on=["column"])
    df_meta["tablename"] = "%s/%s" % (method, new_df_file)
    logger_info.write("Writing Preprocess Data:(%s)..." % new_df_file)
    logger_info.write(new_df.head())
    if sv_test:
        redshif.write(data_frame=new_df, routine_name=method, table_name=new_df_file,
                      bucketname=bucket_name)
    else:
        new_df.to_csv("feat_pp_%s" % new_df_file)
    logger_info.write("Reading meta data:(%s)... " % new_df_file)
    logger_info.write(df_meta)
    old_construct = metadata.construct(routine_name=method, table_name=new_df_file)
    logger_info.write("--Redshift--")
    logger_info.write(old_construct)
    old_construct = pd.merge(old_construct[reds_columns], df_meta, how="left", on=["tablename", "column"])
    logger_info.write("Writing meta data:(%s)... " % new_df_file)
    logger_info.write(old_construct)
    metadata.write(old_construct, routine_name=method, table_name=new_df_file)

    if sv_test:
        del new_df
    del old_construct
    del df_meta
    del dic_type
del df_dic

# Del & Reload new_df_dic
if sv_test:
    file = "feat_adm_math"
    new_df = redshif.read(table_name=file, routine_name=feature_extraction_routine[1])

# FEATURE SELECTION
model = 'rf'
if model == 'rf':
    clf = RandomForestClassifier(n_jobs=-1, max_depth=60, n_estimators=100, class_weight='balanced', )
elif model == 'ln':
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
elif model == 'et':
    clf = ExtraTreesClassifier()
elif model == 'xgb':
    clf = XGBClassifier(learning_rate=1.0)
else:
    raise Exception("No model: %s" % model)

feat_selector = BorutaPy(clf, n_estimators=1000, verbose=2, random_state=1, alpha=0.00001, max_iter=200)
new_df = feature_selection.run(df=new_df, method=feature_selection_routine, select_x=list(new_df.columns),
                               selector=feat_selector, rank=2)
if sv_test:
    original_df = redshif.read(table_name="merge", routine_name="adm")
intersect_col = list(set(new_df.columns) & set(original_df.columns))
new_df = new_df.drop(intersect_col, axis=1)
new_df = pd.concat([original_df, new_df], axis=1)

# Write selected features
if sv_test:
    redshif.write(data_frame=new_df, routine_name=feature_selection_routine,
                  table_name="selected_%s" % feature_selection_routine, bucketname=bucket_name)
else:
    new_df.to_csv("selected.csv")
