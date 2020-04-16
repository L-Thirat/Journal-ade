class characterizer():
    def __init__(self, name, data_type, function):
        self.name = name
        self.data_type = data_type
        self.function = function

    def check_type(self, series):
        dt = str(series.dtype)
        if self.data_type == 'general':
            return True
        elif self.data_type == 'numeric':
            return ('int' in dt) or ('float' in dt)
        elif self.data_type == 'datetime':
            if 'time' in dt:
                return True
            else:
                return False
        else:
            return True

    def characterize(self, series):
        if self.check_type(series):
            output = self.function(series)
            return output
        else:
            return None


# general function
n_unique_value_ch = characterizer('cardinality', 'general', lambda s: len(s.value_counts()))
data_number_ch = characterizer('data_number', 'general', lambda s: len(s))


def data_sample(data):
    """Show sample of data"""
    value_top = []
    count_unique_value = len(data.value_counts())

    # List at most 5 most common values
    value_head = data.value_counts().head()
    for value_head_i in range(0, len(value_head)):
        index = value_head.index.tolist()[value_head_i]
        value = value_head.values.tolist()[value_head_i]
        value_top.append((index, value))
    return value_top


data_sample_ch = characterizer('data_sample', 'general', data_sample)


def percent_missing_value(data):
    """Show percentage of missing value"""
    data_number = len(data)
    check_null = data.isnull().value_counts()
    index = check_null.index.tolist()
    value = check_null.values.tolist()
    number_of_null = 0
    for i in range(0, len(index)):
        if (index[i] == True):
            number_of_null = value[i]
    percent_missing_value = (number_of_null / data_number) * 100
    return percent_missing_value


percent_missing_value_ch = characterizer('percent_missing', 'general', percent_missing_value)

# numarical function
mean_ch = characterizer('mean', 'numeric', lambda s: s.mean())
std_ch = characterizer('std', 'numeric', lambda s: s.std())
median_ch = characterizer('median', 'numeric', lambda s: s.median())
mode_ch = characterizer('mode', 'numeric', lambda s: s.mode()[0])
percentile_25_percent_ch = characterizer('percentile25%', 'numeric', lambda s: s.quantile(.25))
percentile_75_percent_ch = characterizer('percentile75%', 'numeric', lambda s: s.quantile(.75))
kurtosis_ch = characterizer('kurtosis', 'numeric', lambda s: s.kurt())
numeric_min_ch = characterizer('numaric_min', 'numeric', lambda s: s.min())
numeric_max_ch = characterizer('numaric_max', 'numeric', lambda s: s.max())

# datetime function
datetime_min_ch = characterizer('datetime_min', 'datetime', lambda s: s.min())
datetime_max_ch = characterizer('datetime_max', 'datetime', lambda s: s.max())


# categoric function
# top
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
    """Check Unique data"""
    data_number = len(data)
    n_unique_value = len(data.value_counts())
    if n_unique_value == data_number:
        return 1
    else:
        return 0


def percent_unique(data):
    """Calculate percentage of unique data"""
    data_number = len(data)
    n_unique_value = len(data.value_counts())
    return n_unique_value / data_number


all_unique_ch = characterizer('all_unique', 'general', all_unique)
percent_unique_ch = characterizer('percent_unique', 'general', percent_unique)
