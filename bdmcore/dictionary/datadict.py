"""
Written by: Jah Chaisangmongkon

datadict is a module in bdmcore package
It is used for data characterization, built based on numpy and pandas

### General Description ###

'Type': type of the data based on pandas.dtypes function
'Primary Type': our best guess what this column variable type is (categorical, numeric, id, etc)
'Secondary Type' : our second best guess what this column variable type is (categorical, numeric, id, etc)
'Description' : output of pandas.describe()
'N Unique Values' : number of unique values in the column
'N Missing Values' : number of missing values in the column

Description for Specific Variable Types

### Numeric Variable ###
'Skewness': check skewness of the variable

### Categorical Variable ###
'Unique Values' : category names

### ID ###
"""
import numpy as np
import pandas as pd
from scipy.stats import skew


class FeatureProperty:
    order = 0
    title = ''
    description = ''
    value = {}

    def characterize(self, series):
        return self.value


class Type(FeatureProperty):
    order = 1
    title = 'Type'
    value = {}

    def characterize(self, series):
        self.value = series.dtypes.name
        return self.value


class Skewness(FeatureProperty):
    order = 1000
    title = 'Skewness'
    value = {}

    def characterize(self, series):

        # check prerequisite
        # it's a numeric function
        # drop NA

        if Type().characterize(series) != "object":
            self.value = skew(series.dropna())
        else:
            self.value = np.nan

        return self.value


class Description(FeatureProperty):
    title = 'Description'
    order = 5

    def characterize(self, series):
        self.value = series.describe()
        return self.value


class NUniqueValues(FeatureProperty):
    title = "N Unique Values"
    order = 3

    def characterize(self, series):
        self.value = series.value_counts().count()
        return self.value


class RatioUniqueValues(FeatureProperty):
    title = "Ratio Unique Values"
    order = 4

    def characterize(self, series):
        if series.count() == 0:
            self.value = None
        else:
            self.value = series.value_counts().count() / series.count()
        return self.value


class RatioMissingValues(FeatureProperty):
    title = 'Ratio Missing Values'
    order = 4

    def characterize(self, series):
        if series.count() == 0:
            self.value = 0
        else:
            self.value = series.isnull().sum() / series.count()
        return self.value


class PrimaryType(FeatureProperty):
    order = 2
    title = 'Primary Type'
    value = {}

    # this is the function we use to categorize variable type
    def characterize(self, series):
        category_names = ['category']
        numeric_names = ['quantity', 'price',
                         '_count', 'num_unique', 'zero_c', 'sum', 'min', 'max', '_devide', '_multi']
        if series.name.lower() in category_names:
            self.value = 'Categorical'
            return self.value
        if series.name.lower() in numeric_names:
            self.value = 'Numeric'
            return self.value
        t = Type().characterize(series)
        nuniq = NUniqueValues().characterize(series)
        if series.count() == 0:
            self.value = 'Unknown'
            return self.value

        # -->non-object
        temp_series_transform = pd.to_numeric(series, errors='coerce')
        count_nan = float(temp_series_transform.isnull().sum()) / temp_series_transform.count()
        if t == "bool":
            self.value = 'Boolean'
        elif (t != "object") or ((count_nan < 0.2) and (temp_series_transform.count() != 0)):
            if 'datetime64[ns]' in t:
                self.value = 'Date/Time'
            elif 'float' in t:
                if float(nuniq) == 2:
                    self.value = 'Boolean'
                else:
                    self.value = 'Numeric'
            else:
                self.value = 'Missing'
                list_format = ["%Y%m%d", "%d%m%Y"]
                for fm in list_format:
                    try:
                        series = pd.to_datetime(series, format=fm)
                        if series.isnull().sum() / series.count() < 0.2:
                            self.value = 'Date/Time'
                    except:
                        pass
                if self.value == 'Missing':
                    if nuniq / series.count() == 1:
                        self.value = 'ID'
                    elif float(nuniq) == 2:
                        self.value = 'Boolean'
                    elif (float(nuniq)) / series.count() < 0.25:
                        self.value = 'Categorical'
                    else:
                        self.value = 'Numeric'
        else:
            # -->TEXT / Date-Time
            self.value = 'Missing'
            list_format = ["%Y-%m-%d", "%d-%m-%Y", "%Y%m%d", "%d%m%Y", "%Y/%m/%d", "%d/%m/%Y"]
            for fm in list_format:
                try:
                    series = pd.to_datetime(series, format=fm)
                    if series.isnull().sum() / series.count() < 0.2:
                        self.value = 'Date/Time'
                except:
                    pass
            if self.value == 'Missing':
                if nuniq / series.count() == 1:
                    self.value = 'ID'
                elif float(nuniq) == 2:
                    self.value = 'Boolean'
                elif (float(nuniq)) / series.count() < 0.25:
                    self.value = 'Categorical'
                else:
                    self.value = "Text"
        return self.value


class SecondaryType(FeatureProperty):
    title = 'Secondary Type'
    order = 1000

    def characterize(self, series):
        self.value = np.nan
        return self.value


class ValueCounts(FeatureProperty):
    order = 1000
    title = 'Value Counts'
    value = {}

    def characterize(self, series):
        if PrimaryType().characterize(series) == "Categorical":
            self.value = series.value_counts()

        return self.value


def characterize_column(series):
    """
    Example usage:
    a = characterize_data(df["SalePrice"])

    This function perform all characterization jobs described in config.
    Returns the following characteristics of the series:

    :param series: a column of pandas dataframe
    :return: list of FeatureProperty objects
    """

    all_props = [cls.__name__ for cls in globals()['FeatureProperty'].__subclasses__()]
    lis = []
    for prop in all_props:
        constructor = globals()[prop]
        instance = constructor()
        result = instance.characterize(series)
        d = (instance.title, result, instance.order)
        lis.append(d)
    lis = sorted(lis, key=lambda x: x[2])
    return lis


def characterize_table(table):
    """
    This function simply run characterize_feature function for all columns in a table
    :param table:
    :return: python dictionary with column name & list of FeatureProperty objects
    """
    col_names = table.columns
    lis = []
    for name in col_names:
        prop = characterize_column(table[name])
        lis.append((name, prop))
    return lis


def generate_feature_report(table):
    # printing table head() 6 columns at a time
    h = table.head().to_string()

    # printing characteristics of each feature
    st = ''
    lis = characterize_table(table)
    feature_chars = {}

    for feat in lis:
        st = st + '\nColumn Name: ' + feat[0] + '\n'
        for prop in feat[1]:
            st = st + prop[0] + ': '
            if '\n' in str(prop[1]):
                st = st + '\n' + str(prop[1]) + '\n'
            else:
                st = st + str(prop[1]) + '\n'
        feature_chars[feat[0]] = feat[1]

    return 'feature_report', {'table_head': h, 'feature_char': st}, feature_chars


if __name__ == "__main__":
    pass
