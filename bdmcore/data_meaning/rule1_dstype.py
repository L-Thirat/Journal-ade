import pandas as pd
import logging

log = logging.getLogger(__file__)


def check_conversion(serie, convert_to):
    try:
        serie.dropna().astype(convert_to)
        return True
    except ValueError:
        return False


def check_column(serie, column, dtype, unique_score, upper_unique_score=0.9, lower_unique_score=0.1):
    if ('id' in column) and (unique_score > upper_unique_score):
        if dtype == float:
            serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        else:
            serie.loc[serie.notnull()] = serie.dropna().astype(str)
        return 'id', serie, 'object'

    if ('id' in column) and (dtype == int):
        serie.loc[serie.notnull()] = serie.dropna().astype(str)
        return 'id', serie, 'object'

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

    if (dtype == object) and (unique_score < lower_unique_score):
        return 'categorical', serie, 'object'

    if (dtype == float) and (unique_score < lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        return 'categorical', serie, 'object'

    if (dtype == float) and (unique_score > lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(float)
        return 'numeric', serie, 'float'

    if (dtype == int) and (unique_score < lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(str)
        return 'categorical', serie, 'object'

    if (dtype == int) and (unique_score >= lower_unique_score):
        serie.loc[serie.notnull()] = serie.dropna().astype(float)
        return 'numeric', serie, 'float'

    if dtype == object:
        try:
            serie.dropna().astype(float)
            serie.loc[serie.notnull()] = serie.dropna().astype(int).astype(float)
            return 'numeric', serie, 'float'
        except ValueError:
            log.info('fail float conversion test')

    if (dtype == object) and (unique_score > upper_unique_score):
        return 'id', serie, 'object'

    if (dtype == object) and (unique_score >= lower_unique_score):
        return 'text', serie, 'object'


def create_dstype(df, mdf):
    # todo check validity of mdf
    mdf = mdf.set_index('column_name')
    df_output = mdf

    if len(df) > 50000:
        df = df.sample(50000)

    for column in df.columns:
        log.info('checking column: {}'.format(column))

        unique_n = mdf.loc[column, 'cardinality']
        unique_score = mdf.loc[column, 'percent_unique']
        dtype = df[column].dtype

        df_output.loc[column, 'dstype'], df[column], df_output.loc[column, 'variable_type'] = check_column(df[column],
                                                                                                           column,
                                                                                                           dtype,
                                                                                                           unique_score)
        log.info('caught as {}'.format(df_output.loc[column, 'dstype']))

    df_output = df_output.reset_index()

    return df_output
