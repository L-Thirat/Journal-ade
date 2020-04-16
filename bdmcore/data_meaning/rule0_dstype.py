import pandas as pd
import logging

log = logging.getLogger(__file__)


def check_column(serie, column, dtype, unique_score, cardinality, upper_unique_score=0.9, lower_unique_score=0.1):
    # detect ids
    if ('id' in column) and (unique_score > upper_unique_score):
        if dtype == float:
            serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        else:
            serie.loc[serie.notnull()] = serie.dropna().astype(str)
        return 'id', serie, 'object'

    if ('id' in column) and (dtype == int):
        serie.loc[serie.notnull()] = serie.dropna().astype(str)
        return 'id', serie, 'object'

    # detect dates
    if dtype == object:
        try:
            if len(serie) >= 100:
                test_series = serie.dropna().iloc[0:100]
            else:
                test_series = serie.dropna()
            pd.to_datetime(test_series)
            return 'date', serie, 'object'
        except ValueError:
            log.info('fail date conversion test')

    # detect string fields that are likely categorical
    if (dtype == object) and (cardinality < 100) and (unique_score < lower_unique_score):
        return 'categorical', serie, 'object'

    # detect float fields that are likely categorical
    if (dtype == float) and (cardinality < 100) and (unique_score < lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        return 'categorical', serie, 'object'

    # detect int fields that are likely categorical
    if (dtype == int) and (cardinality < 100) and (unique_score < lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        return 'categorical', serie, 'object'

    # all other float fields should now be regarded as numeric
    if dtype == float:
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(float)
        return 'numeric', serie, 'float'

    # all other int fields should now be regarded as numeric
    if dtype == int:
        serie.loc[serie.notnull()] = serie.dropna().astype(float)
        return 'numeric', serie, 'float'

    # test whether str fields can be converted to float and numeric
    if dtype == object:
        try:
            serie.dropna().astype(float)
            serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(float)
            return 'numeric', serie, 'float'
        except ValueError:
            log.info('fail float conversion test')

    if (dtype == object) and (unique_score < lower_unique_score):
        return 'categorical', serie, 'object'

    # highly unique string should become ids
    if (dtype == object) and (unique_score > upper_unique_score):
        return 'id', serie, 'object'

    # other str should become text
    if (dtype == object) and (unique_score >= lower_unique_score):
        return 'text', serie, 'object'

    log.info("Cannot classify with (dtype, unique_score, cardinality) = {}".format((dtype, unique_score, cardinality)))
    return None, serie, None


def create_dstype(df, mdf):
    # todo check validity of mdf
    mdf = mdf.set_index('column_name')
    df_output = mdf

    if len(df) > 50000:
        df = df.sample(50000)

    for column in df.columns:
        log.info('checking column: {}'.format(column))

        cardinality = mdf.loc[column, 'cardinality']
        unique_score = mdf.loc[column, 'percent_unique']
        dtype = df[column].dtype

        df_output.loc[column, 'dstype'], df[column], df_output.loc[column, 'variable_type'] = check_column(
            df[column], column, dtype, unique_score, cardinality)
        log.info('caught as {}'.format(df_output.loc[column, 'dstype']))

    df_output = df_output.reset_index()

    return df_output
