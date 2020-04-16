import numpy as np
import string
from dateutil.parser import parse


class characterizer():
    def __init__(self, name, data_type, function):
        self.name = name
        self.data_type = data_type
        self.function = function

    def characterize(self, series):
        return self.function(series)


class converter(characterizer):
    def __init__(self):
        self.data_digit = ""
        self.data_character = ""
        self.data_date = ""
        self.data_date1 = ""
        self.data_specialchar_only = ""
        self.data_specialchar = ""
        self.data_capitalletter = ""
        self.data_smallletter = ""
        self.data_digitplus = ""

    def character_only(self, data):

        self.data_character = ""

        try:
            if data.isalpha():
                self.data_character = True
            else:
                self.data_character = False
        except Exception:
            self.data_character = False

        return self.data_character

    def digit_only(self, data):

        self.data_digit = ""

        try:
            if data.replace('.', '').isdigit():
                self.data_digit = True
            else:
                self.data_digit = False
        except Exception:
            self.data_digit = False

        return self.data_digit

    def capitalletters_only(self, data):

        self.data_capitalletter = ""

        try:
            if data.isupper():
                self.data_capitalletter = True
            else:
                self.data_capitalletter = False
        except Exception:
            self.data_capitalletter = False

        return self.data_capitalletter

    def smallletters_only(self, data):

        self.data_smallletter = ""

        try:
            if data.islower():
                self.data_smallletter = True
            else:
                self.data_smallletter = False
        except Exception:
            self.data_smallletter = False
        return self.data_smallletter

    def specialcharacter_only(self, data):
        """Check if all of string are special character

        """
        self.data_specialchar_only = ""
        count = 0
        invalidChars = set(string.punctuation.replace("_", ""))

        for i in data:
            if i in invalidChars:
                count += 1
                if count == len(data):
                    self.data_specialchar_only = True
            else:
                self.data_specialchar_only = False
                break

        return self.data_specialchar_only

    def specialcharacter(self, data):
        """Check if special character in data

        """
        self.data_specialchar = ""
        invalidChars = set(string.punctuation.replace("_", ""))

        if any(char in invalidChars for char in data):
            self.data_specialchar = True
        else:
            self.data_specialchar = False

        return self.data_specialchar

    def is_date(self, data):

        self.data_date = ""
        try:
            parse(data)
            self.data_date = True
        except ValueError:
            self.data_date = False
        except OverflowError:
            self.data_date = False

        return self.data_date

    @staticmethod
    def count_length(data):

        return len(data)

    @staticmethod
    def count_number(data):
        number = "0123456789"
        count = 0

        for i in data:
            if i in number:
                count += 1

        return count

    @staticmethod
    def count_cazpitalletters(data):
        count = 0

        for i in data:
            try:
                if i.isupper():
                    count += 1
            except Exception:
                count = 0

        return count

    @staticmethod
    def count_smallletters(data):
        count = 0

        for i in data:
            try:
                if i.islower():
                    count += 1
            except Exception:
                count = 0

        return count

    @staticmethod
    def count_specialcharacter(data):
        """Count special character in data

        """
        invalidChars = set(string.punctuation.replace("_", ""))
        count = 0

        for i in data:
            if i in invalidChars:
                count += 1

        return count

    @staticmethod
    def count_dot(data):
        symbol = "."
        count = 0

        for i in data:
            if i in symbol:
                count += 1

        return count

    @staticmethod
    def count_forwardslash(data):
        symbol = "/"
        count = 0

        for i in data:
            if i in symbol:
                count += 1

        return count

    @staticmethod
    def count_dash(data):
        symbol = "-"
        count = 0

        for i in data:
            if i in symbol:
                count += 1

        return count

    @staticmethod
    def count_comma(data):
        symbol = ","
        count = 0

        for i in data:
            if i in symbol:
                count += 1

        return count

    def digit_comma(self, data):

        symbol = ","
        count = 0

        self.data_digitplus = ""

        for i in data:
            if i in symbol:
                count += 1
                if (data.replace(symbol, '')).isdigit():
                    self.data_digitplus = True
                else:
                    self.data_digitplus = False
                    break
            else:
                count += 1
                if count == len(data):
                    self.data_digitplus = False

        return self.data_digitplus


class numeric():

    @staticmethod
    def average(data):
        return np.mean(data)

    @staticmethod
    def maximum(data):
        return np.max(data)

    @staticmethod
    def minimum(data):
        return np.min(data)

    @staticmethod
    def std_dev(data):
        return np.std(data)


class boolean():

    @staticmethod
    def all_true(data):

        return all(data) is True

    @staticmethod
    def all_false(data):

        return all(data) is False

    @staticmethod
    def any_true(data):

        return any(data) is True

    @staticmethod
    def any_false(data):

        return any(data) is False

    @staticmethod
    def true_proportion(data):
        count_ture = data.isin([True]).sum()

        if count_ture == 0:
            t_p = 0
        else:
            t_p = count_ture / float(len(data))

        return t_p

    @staticmethod
    def false_proportion(data):
        count_false = data.isin([False]).sum()

        if count_false == 0:
            t_f = 0
        else:
            t_f = count_false / float(len(data))

        return t_f


# -------------------- general function -----------------------------
n_unique_value_ch = characterizer('n_unique_value', 'general', lambda s: len(s.value_counts()))
data_number_ch = characterizer('data_number', 'general', lambda s: len(s))


def data_sample(data):
    value_top = []

    # List at most 5 most common values
    value_head = data.value_counts().head()
    for value_head_i in range(0, len(value_head)):
        index = value_head.index.tolist()[value_head_i]
        value = value_head.values.tolist()[value_head_i]
        value_top.append((index, value))
    return value_top


data_sample_ch = characterizer('data_sample', 'general', data_sample)


# percentage null data
def percent_missing_value(data):
    data_number = len(data)
    check_null = data.isnull().value_counts()
    index = check_null.index.tolist()
    value = check_null.values.tolist()
    number_of_null = 0
    for i in range(0, len(index)):
        if index[i]:
            number_of_null = value[i]
    percent_missing_value = (number_of_null / data_number) * 100
    return percent_missing_value


percent_missing_value_ch = characterizer('percent_missing_value', 'general', percent_missing_value)

# -----------------------numarical function---------------------------
mean_ch = characterizer('mean', 'numeric', lambda s: s.mean())
std_ch = characterizer('std', 'numeric', lambda s: s.std())
median_ch = characterizer('median', 'numeric', lambda s: s.median())
mode_ch = characterizer('mode', 'numeric', lambda s: s.mode()[0])
percentile_25_percent_ch = characterizer('percentile25%', 'numeric', lambda s: s.quantile(.25))
percentile_75_percent_ch = characterizer('percentile75%', 'numeric', lambda s: s.quantile(.75))
kurtosis_ch = characterizer('kurtosis', 'numeric', lambda s: s.kurt())
numeric_min_ch = characterizer('numaric_min', 'numeric', lambda s: s.min())
numeric_max_ch = characterizer('numaric_max', 'numeric', lambda s: s.max())

# ----------------------datetime function-------------------------------
datetime_min_ch = characterizer('datetime_min', 'datetime', lambda s: s.min())
datetime_max_ch = characterizer('datetime_max', 'datetime', lambda s: s.max())


# ----------------------categoric function-------------------------------
def top_5(data):
    data_number = len(data)
    data_count = data.value_counts()
    data_top = data_count.head()
    top_5 = []

    top_index = data_top.index.tolist()
    for i in range(0, len(top_index)):
        top_5.append(top_index[i])
    return top_5


def freq_top_5(data):
    data_number = len(data)
    data_count = data.value_counts()
    data_top = data_count.head()
    freq_top_5 = []

    top_value = data_top.values.tolist()
    for i in range(0, len(top_value)):
        percent_value = top_value[i] * 100 / data_number
        percent_value = format(percent_value, '.2f')
        freq_top_5.append(percent_value)
    return freq_top_5


def bottom_5(data):
    data_number = len(data)
    data_count = data.value_counts()
    data_bottom = data_count.tail()
    bottom_5 = []

    bottom_index = data_bottom.index.tolist()
    for i in range(0, len(bottom_index)):
        bottom_5.append(bottom_index[i])
    return bottom_5


def freq_bottom_5(data):
    data_number = len(data)
    data_count = data.value_counts()
    data_bottom = data_count.tail()
    freq_bottom_5 = []

    bottom_value = data_bottom.values.tolist()
    for i in range(0, len(bottom_value)):
        percent_value = bottom_value[i] * 100 / data_number
        percent_value = format(percent_value, '.2f')
        freq_bottom_5.append(percent_value)
    return freq_bottom_5


top_5_ch = characterizer('top_5', 'categoric', top_5)
freq_top_5_ch = characterizer('freq_top_5', 'categoric', freq_top_5)
bottom_5_ch = characterizer('bottom_5', 'categoric', bottom_5)
freq_bottom_5_ch = characterizer('freq_bottom_5', 'categoric', freq_bottom_5)


def all_unique(data):
    data_number = len(data)
    n_unique_value = len(data.value_counts())
    if n_unique_value == data_number:
        return 1
    else:
        return 0


all_unique_ch = characterizer('all_unique', 'ID', all_unique)


def all_converter(data):
    function_characterizer = dir(characterizer)
    function_characterizer = [i for i in function_characterizer if "_" != i[0] and "_" != i[1] in i]
    function_converter = dir(converter)
    function_converter = [i for i in function_converter if "_" != i[0] and "_" != i[1] in i]
    function_converter = list(set(function_converter) - set(function_characterizer))
    function_numeric = dir(numeric)
    function_numeric = [i for i in function_numeric if "_" != i[0] and "_" != i[1] in i]
    function_boolean = dir(boolean)
    function_boolean = [i for i in function_boolean if "_" != i[0] and "_" != i[1] in i]

    a = {}

    for x in range(0, len(function_converter)):
        result = data.apply(getattr(converter(), function_converter[x]))

        # Check Type 
        # Boolean
        if result[0].dtype == bool:
            for booleans in function_boolean:
                a[booleans + "-" + function_converter[x]] = getattr(boolean(), booleans)(result)
        # Numeric
        else:
            for numerics in function_numeric:
                a[numerics + "-" + function_converter[x]] = getattr(numeric(), numerics)(result)

    return a


all_converter_ch = characterizer('all_converter', 'general', all_converter)
