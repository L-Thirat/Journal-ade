from bdmcore.dictionary.type import declare_variable_type
from bdmcore.dictionary.type import declare_data_type


def run(df, method, drop_text=True):
    """Confirm data type pipeline"""
    global variable_types
    if method == "adm":
        df, variable_types = declare_variable_type(df)
        df, data_types = declare_data_type(df, drop_text)
    else:
        raise Exception("No method: %s" % method)
    return df, variable_types, data_types
