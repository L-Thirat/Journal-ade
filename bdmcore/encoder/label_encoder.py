from sklearn import preprocessing
from bdmcore.dsproject import dsProject
from bdmcore.util import setup_logging
import pandas as pd

# setting
setup_logging()
dsp = dsProject(name="kdd2015_dropout")
dsp.define_target_label(['enrollment_id'], ['label'])
labenc = preprocessing.LabelEncoder()

# load data
df1 = dsp.load_data_table('log_train.csv', 'data')
df2 = dsp.load_data_table('log_validate.csv', 'data')
df3 = dsp.load_data_table('log_test.csv', 'data')
df = pd.concat([df1, df2])
df = pd.concat([df, df3])

# label encoding
cat = ['source', 'event', 'object']
for column in cat:
    original = column
    new = 'hash_' + column
    df[new] = df[column].astype(object)
    try:
        df[new] = labenc.fit_transform(df[column])
    except Exception as e:
        print(e)
        null = df[column][df[column].isnull()]
        notnull = (df[column][~df[column].isnull()])
        df[new][~df[column].isnull()] = labenc.fit_transform(notnull)
        df[new] = pd.concat([null, df[new][~df[column].isnull()]], axis=0, ignore_index=False)
    df_out = df[[new, column]].copy().drop_duplicates()
    dsp.write_feature(df_out, 'hash_%s' % column)
