import datetime
import pandas as pd
import numpy as np


class IsNull():
    """For each value of base feature, return true if value is null"""
    # name = "is_null"
    # input_types = [Variable]
    # return_type = Boolean

    def get_function(self):
        return lambda array: pd.isnull(pd.Series(array))


class Absolute():
    """Absolute value of base feature"""
    # name = "absolute"
    # input_types = [Numeric]
    # return_type = Numeric

    def get_function(self):
        return lambda array: np.absolute(array)


class DatetimeUnitBasePrimitive():
    """Transform Datetime feature into time or calendar units (second/day/week/etc)"""
    name = None
    # input_types = [Datetime]
    # return_type = Ordinal

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.DatetimeIndex(array))

class TimedeltaUnitBasePrimitive():
    """Transform Timedelta features into number of time units (seconds/days/etc) they encompass"""
    name = None
    # input_types = [Timedelta]
    # return_type = Numeric

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.TimedeltaIndex(array))



class Day(DatetimeUnitBasePrimitive):
    name = "day"


class Days(TimedeltaUnitBasePrimitive):
    name = "days"


class Hour(DatetimeUnitBasePrimitive):
    name = "hour"


class Hours(TimedeltaUnitBasePrimitive):
    name = "hours"

    def get_function(self):
        return lambda array: pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 3600.


class Second(DatetimeUnitBasePrimitive):
    name = "second"


class Seconds(TimedeltaUnitBasePrimitive):
    name = "seconds"


class Minute(DatetimeUnitBasePrimitive):
    name = "minute"


class Minutes(TimedeltaUnitBasePrimitive):
    name = "minutes"

    def get_function(self):
        return lambda array: pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 60.


class Week(DatetimeUnitBasePrimitive):
    name = "week"


class Weeks(TimedeltaUnitBasePrimitive):
    name = "weeks"

    def get_function(self):
        return lambda array: pd_time_unit("days")(pd.TimedeltaIndex(array)) / 7.


class Month(DatetimeUnitBasePrimitive):
    name = "month"


class Months(TimedeltaUnitBasePrimitive):
    name = "months"

    def get_function(self):
        return lambda array: pd_time_unit("days")(pd.TimedeltaIndex(array)) * (12. / 365)


class Year(DatetimeUnitBasePrimitive):
    name = "year"


class Years(TimedeltaUnitBasePrimitive):
    name = "years"

    def get_function(self):
        return lambda array: pd_time_unit("days")(pd.TimedeltaIndex(array)) / 365


class Weekend():
    """Transform Datetime feature into the boolean of Weekend"""
    name = "is_weekend"
    # input_types = [Datetime]
    # return_type = Boolean

    def get_function(self):
        return lambda df: pd_time_unit("weekday")(pd.DatetimeIndex(df)) > 4


class Weekday(DatetimeUnitBasePrimitive):
    name = "weekday"


class TimeSince():
    """For each value of the base feature, compute the timedelta between it and a datetime"""
    name = "time_since"
    # input_types = [[DatetimeTimeIndex], [Datetime]]
    # return_type = Timedelta
    # uses_calc_time = True

    def get_function(self):
        def pd_time_since(array, time):
            if time is None:
                time = datetime.now()
            return (time - pd.DatetimeIndex(array)).values
        return pd_time_since


class DaysSince():
    """For each value of the base feature, compute the number of days between it and a datetime"""
    name = "days_since"
    # input_types = [DatetimeTimeIndex]
    # return_type = Numeric
    # uses_calc_time = True

    def get_function(self):
        def pd_days_since(array, time):
            if time is None:
                time = datetime.now()
            return pd_time_unit('days')(time - pd.DatetimeIndex(array))
        return pd_days_since


class Percentile():
    name = 'percentile'
    # input_types = [Numeric]
    # return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


def pd_time_unit(time_unit):
    def inner(pd_index):
        return getattr(pd_index, time_unit).values
    return inner
