from bdmcore.dsproject import Config
from bdmcore.dsproject import Metadata
from bdmcore.dsproject import RedshiftConnector
from bdmcore.dsproject import Logger
from bdmcore import feature_selection

# Project setting
ProjectName = "fb_recruitingiv"
dtype_routine = "adm"
feature_extraction_routine = ["adm_extract", "adm_math"]
feature_selection_routine = "boruta"
master_id = ["bidder_id"]
label = ["outcome"]
data_files = ['train_csv', 'bids_csv']
set_val = {"files": data_files, "mid": master_id, "label": label}
feature_selection.setup(set_val)

# Config
bucket_name = "fb_recruitingiv"
conf = Config(ProjectName)
redshif = RedshiftConnector(config=conf)
metadata = Metadata(conf, redshif)
logger_info = Logger(ProjectName, conf)
reds_columns = ['tablename', 'column', 'type']

# read data
df = redshif.read("h_train_csv")
print(df.head())

metadata.read()
meta = metadata.construct(routine_name="adm_extract", table_name="feat_adm_extract")
meta.to_csv("feat_adm_extract.csv")

meta = redshif.read(table_name="feat_adm_extract", routine_name="adm_extract")
meta.to_csv("feat_adm_extract.csv")
