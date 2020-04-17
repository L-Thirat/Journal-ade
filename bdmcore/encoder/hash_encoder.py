from bdmcore.dsproject import dsProject
from bdmcore.util import setup_logging
import pandas as pd

# setting
setup_logging()
dsp = dsProject(name="kdd2015_dropout")
dsp.define_target_label(['enrollment_id'],['label'])

# load hash data
hash_f = dsp.load_data_tables('features',sub='hash_',nrows=1)

col_hash=[]
for file in hash_f:
    col_hash.append(hash_f[file].columns[1])

# load preprocess data
dsp.feature_tables = {}
all_df = dsp.load_data_tables('features',sub='pre_t')

# hash encoding
for file in all_df:
    for col in all_df[file]:
        if col in col_hash:
            name_h = "hash_%s.csv"%col
            df_h = dsp.load_data_table(name_h,'features')
            all_df[file] = pd.merge(all_df[file],df_h,how='left',on=col)
            all_df[file][col] = all_df[file][('hash_%s'%col)]
            del all_df[file][('hash_%s'%col)]
    dsp.write_feature((all_df[file]),'h_%s'%file[:-4])

