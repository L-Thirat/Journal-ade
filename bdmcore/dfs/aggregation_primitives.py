import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import skew


class ZeroCount:
    """Count number of zero"""
    name = "zero_c"
    # input_types =  ['Numeric']
    return_type = 'Numeric'
    # stack_on_self = False
    # stack_on_exclude = ['Count']

    @staticmethod
    def get_function():
        """generate zero count function

        :return: zero count function
        """
        def zero_count(x):
            return (len(list(x)) - np.count_nonzero(x.values))/len(list(x))
        return zero_count


class Count:
    """Counts the number of non null values."""
    name = "count"
    # input_types =  [[Index], [Variable]]
    return_type = 'Numeric'
    # stack_on_self = False
    # default_value = 0

    def __init__(self, count_null=False):
        self.count_null = count_null

    def get_function(self):
        """generate nan count function

        :return: nan count function
        """
        def func(values, count_null=self.count_null):
            if len(values) == 0:
                return 0

            if count_null:
                values = values.fillna(0)
            return values.count()
        return func


class Sum:
    """Aggregate sum"""
    name = "sum"
    # input_types =  [Numeric]
    return_type = 'Numeric'
    # stack_on_self = False
    # stack_on_exclude = [Count]

    @staticmethod
    def get_function():
        """generate aggregate sum function

        :return: aggregate sum function
        """
        def sum_func(x):
            return np.nan_to_num(x.values).sum(dtype=np.float)
        return sum_func


class Mean:
    """Aggregate mean"""
    name = "mean"
    # input_types =  [Numeric]
    return_type = 'Numeric'

    @staticmethod
    def get_function():
        """generate aggregate mean function

        :return: aggregate mean function
        """
        return np.nanmean


class Mode:
    """Finds the most common element in a categorical feature."""
    name = "mode"
    # input_types =  [Discrete]
    return_type = None

    @staticmethod
    def get_function():
        """generate aggregate mode function

        :return: aggregate mode function
        """
        def pd_mode(x):
            if x.mode().shape[0] == 0:
                return np.nan
            return x.mode().iloc[0]
        return pd_mode


class Min:
    """Finds the minimum non-null value of a numeric feature."""
    name = "min"
    # input_types =  [Numeric]
    return_type = None
    # # max_stack_depth = 1
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate aggregate min function

        :return: aggregate min function
        """
        return np.min


class Max:
    """Finds the maximum non-null value of a numeric feature."""
    name = "max"
    # input_types =  [Numeric]
    return_type = None
    # # max_stack_depth = 1
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate aggregate max function

        :return: aggregate max function
        """
        return np.max


class NUnique:
    """Returns the number of unique categorical variables"""
    name = "num_unique"
    # input_types = [Discrete]
    return_type = 'Numeric'
    # max_stack_depth = 1
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate number of unique categorical count function

        :return: number of unique categorical count function
        """
        return lambda x: x.nunique()


class PercentTrue:
    """Finds the percent of 'True' values in a boolean feature."""
    name = "percent_true"
    # input_types =  [Boolean]
    return_type = 'Numeric'
    # max_stack_depth = 1
    # stack_on = []
    # stack_on_exclude = []

    @staticmethod
    def get_function():
        """generate percent of true value calculate function

        :return: percent of true value calculate function
        """
        def percent_true(x):
            if len(x) == 0:
                return np.nan
            return np.nan_to_num(x.values).sum(dtype=np.float) / len(x)
        return percent_true


class AvgTimeBetween:
    """
    Computes the average time between consecutive events
    using the time index of the entity

    Note: equivalent to Mean(Diff(time_index)), but more performant
    """
    name = "avg_time_between"
    # input_types =  [DatetimeTimeIndex]
    return_type = 'Numeric'
    # # max_stack_depth = 1

    @staticmethod
    def get_function():
        """generate average time in range calculate function

        :return: average time in range calculate function
        """
        def pd_avg_time_between(x):
            x = x.dropna()
            x = pd.to_datetime(x)
            if x.shape[0] < 2:
                return np.nan
            avg = int(((x.max() - x.min()) / float(len(x) - 1)).days)
            return avg
        return pd_avg_time_between


class Median:
    """Finds the median value of any feature with well-ordered values."""
    name = "median"
    # input_types =  [Numeric]
    return_type = None
    # # max_stack_depth = 2

    @staticmethod
    def get_function():
        """generate aggregate median function

        :return: aggregate median function
        """
        return np.median


class Skew:
    """Computes the skewness of a data set.

    For normally distributed data, the skewness should be about 0. A skewness
    value > 0 means that there is more weight in the left tail of the
    distribution.
    """
    name = "skew"
    # input_types =  [Numeric]
    return_type = 'Numeric'
    # stack_on = []
    # stack_on_self = False
    # # max_stack_depth = 1

    @staticmethod
    def get_function():
        """generate aggregate skew function

        :return: aggregate skew function
        """
        return skew


class Std:
    """Finds the standard deviation of a numeric feature ignoring null values."""
    name = "std"
    # input_types =  [Numeric]
    return_type = 'Numeric'
    # # max_stack_depth = 2
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate aggregate std function

        :return: aggregate std function
        """
        return np.nanstd


class Last:
    """Returns the last value"""
    name = "last"
    # input_types =  [Variable]
    return_type = None
    # stack_on_self = False
    # # max_stack_depth = 1

    @staticmethod
    def get_function():
        """generate finding last value function

        :return: finding last value function
        """
        def pd_last(x):
            return x.iloc[-1]
        return pd_last


class Any:
    """Test if any value is True"""
    name = "any"
    # input_types =  [Boolean]
    return_type = 'Boolean'
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate checking if any value is True function

        :return: checking if any value is True function
        """
        return np.any


class All:
    """Test if all values are True"""
    name = "all"
    # input_types =  [Boolean]
    return_type = 'Boolean'
    # stack_on_self = False

    @staticmethod
    def get_function():
        """generate checking if all value is True function

        :return: checking if all value is True function
        """
        return np.all


class Trend:
    """Calculates the slope of the linear trend of variable overtime"""
    name = "trend"
    # input_types =  [Numeric, DatetimeTimeIndex]
    return_type = 'Numeric'

    @staticmethod
    def get_function():
        """generate trend calculate function

        :return: trend calculate function
        """
        def pd_trend(y):
            df = (y.reset_index()).dropna()
            df.columns = ['x', 'y']
            if df.shape[0] <= 2:
                return np.nan
            if isinstance(df['x'].iloc[0], (datetime, pd.Timestamp)):
                x = convert_datetime_to_floats(df['x'])
            else:
                x = df['x'].values

            if isinstance(df['y'].iloc[0], (datetime, pd.Timestamp)):
                y = convert_datetime_to_floats(df['y'])
            elif isinstance(df['y'].iloc[0], (timedelta, pd.Timedelta)):
                y = convert_timedelta_to_floats(df['y'])
            else:
                y = df['y'].values

            x = x - x.mean()
            y = y - y.mean()

            # prevent divide by zero error
            if len(np.unique(x)) == 1:
                return 0

            # consider scipy.stats.linregress for large n cases
            coefficients = np.polyfit(x, y, 1)

            return coefficients[0]
        return pd_trend


def convert_datetime_to_floats(x):
    """Convert datetime data to Numeric data

    :param x: data
    :return data converted
    """
    first = int(x.iloc[0].value * 1e-9)
    x = pd.to_numeric(x).astype(np.float64).values
    dividend = find_dividend_by_unit(first)
    x *= (1e-9 / dividend)
    return x


def convert_timedelta_to_floats(x):
    """Convert timedata data to Numeric data

    :param x: data
    :return data converted
    """
    first = int(x.iloc[0].total_seconds())
    dividend = find_dividend_by_unit(first)
    x = pd.TimedeltaIndex(x).total_seconds().astype(np.float64) / dividend
    return x


def find_dividend_by_unit(time):
    """Finds whether time best corresponds to a value in days, hours, minutes, or seconds

    :param time: time
    :return range of time
    """
    for dividend in [86400., 3600., 60.]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1
