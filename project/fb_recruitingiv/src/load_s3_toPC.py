import sys
sys.path.append("/home/ubuntu/core")
from bdmcore.dsproject import config
from bdmcore.dsproject import metadata
from bdmcore.dsproject import redshift_connector
from bdmcore.dsproject import logger
from bdmcore import feature_selection


ProjectName = "fb_recruitingiv"
dtype_routine = "adm"
feature_extraction_routine = ["adm_extract","adm_math"]
feature_selection_routine = "boruta"
master_id = ["bidder_id"]
label = ["outcome"]
data_files = ['train_csv','bids_csv']
set_val = {"files":data_files,"mid":master_id,"label":label}
feature_selection.setup(set_val)

# Config
bucket_name = "fb_recruitingiv"
conf = config(ProjectName)
redshif = redshift_connector(config=conf)
metadata = metadata(conf,redshif)
logger_info = logger(ProjectName,conf)
reds_columns = ['tablename','column','type']

df = redshif.read("h_micro_bids_csv")
df = redshif.read("h_train_csv")
print(df.head())

meta = metadata.read()
meta = metadata.construct(routine_name="adm_extract",table_name="feat_adm_extract")
meta.to_csv("feat_adm_extract.csv")

meta = redshif.read(table_name="feat_adm_extract",routine_name="adm_extract")
meta.to_csv("feat_adm_extract.csv")
