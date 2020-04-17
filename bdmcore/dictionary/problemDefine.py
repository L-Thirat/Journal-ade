from itertools import combinations
from bdmcore.dictionary.datadict import RatioUniqueValues
import pandas as pd


def entity_discovery(data_tables):
    """
    use self.tables to automatically map out data science problem

    Logic:
    Each table represent an event type, table name = event name
    Some columns are entity id's:
        - columns with unique id's
        - columns that appear in more than one table
    Some columns are entity properties:
        - columns that have same values for same entity id's
    Some columns are event properties:
        - columns that can be found in only one table

    :param
    :return: entity dataframe, event dataframe
    """

    entity = {}
    temp_col = []
    intersect_col = []
    for tab_name, tab in data_tables.items():
        entity[tab_name] = pd.DataFrame(columns=['main entity', 'entity', 'entity property', 'event property'],
                                        index=tab.columns)
        entity[tab_name] = entity[tab_name].fillna(0)
        if len(data_tables) < 2:
            for col in list(tab.columns):
                n = RatioUniqueValues().characterize(tab[col])
                if n == 1.0:
                    intersect_col.append(col)
        else:
            for col in list(tab.columns):
                if col in temp_col:
                    intersect_col.append(col)
            temp_col = temp_col + (list(tab.columns))

    # loop through all columns of a table and find columns with unique ids, register these as entities
    for tab_name, tab in data_tables.items():
        for column_name in tab.columns:
            n = RatioUniqueValues().characterize(tab[column_name])
            if column_name in intersect_col:
                if (n == 1.0) and (1 not in list(entity[tab_name]['main entity'])):
                    entity[tab_name]['main entity'].loc[column_name] = 1
                else:
                    entity[tab_name]['entity'].loc[column_name] = 1

    # loop through entities
    # find columns that are property of that entity
    # if not that column is an event property
    for tab_name, tab in data_tables.items():
        for column_name in tab.columns:
            if entity[tab_name]['main entity'].loc[column_name]:
                for column_name1 in tab.columns:
                    if column_name1 != column_name:
                        temp = tab.groupby(column_name).agg({column_name1: 'nunique'})
                        if int(temp.sum()) / int(temp.count()) == 1.0:
                            if entity[tab_name]['entity property'].loc[column_name1]:
                                entity[tab_name]['entity property'].loc[column_name1] += ',' + column_name
                            else:
                                entity[tab_name]['entity property'].loc[column_name1] = column_name

        for column_name in tab.columns:
            if not entity[tab_name]['entity'].loc[column_name] and not \
                    entity[tab_name]['entity property'].loc[column_name] and not \
                    entity[tab_name]['main entity'].loc[column_name]:
                entity[tab_name]['event property'].loc[column_name] = 1
        entity[tab_name].index.name = 'column'
        entity[tab_name] = entity[tab_name].reset_index()
    return entity


def data_relationship_discovery(data_tables, entity):
    """Create relation table"""
    # initiate table relationship map
    data_files = data_tables.keys()

    combs = combinations(data_files, 2)
    table_rel = {}
    for comb in combs:
        tab1 = data_tables[comb[0]]
        tab2 = data_tables[comb[1]]
        tmp = pd.DataFrame(columns=tab1.columns, index=tab2.columns)
        tmp = tmp.fillna(0)
        table_rel[comb[0] + '_' + comb[1]] = tmp

    # loop through tables and find columns that appear in more than one tables (same set of unique values)
    # register these as entities
    for comb in combs:
        tab1 = data_tables[comb[0]]
        tab2 = data_tables[comb[1]]
        inter = set(tab1.columns).intersection(set(tab2.columns))

        for i in inter:
            tab1_unique = tab1[i].unique().tolist()
            tab2_unique = tab2[i].unique().tolist()
            values = set(tab1_unique).intersection(tab2_unique)
            if len(values):  # len(values) >= len(tab1_unique) or len(values) >= len(tab2_unique):
                entity[comb[0]]['entity'].loc[i] = 1
                entity[comb[1]]['entity'].loc[i] = 1
                table_rel[comb[0] + '_' + comb[1]][i].loc[i] = 's'
    return table_rel, entity


def find_main_id(entity):
    """Find Main ID which use to predict"""
    # objective :: find main table that have main entity
    if 1 in list(entity["main entity"]):
        main_id = str(list(entity[entity["main entity"] == 1]['column']))[2:-2]
    else:
        main_id = None
    return main_id


def find_main_table(entity_dic, master_id):
    """Find Main Table"""
    main_tables = []
    for file in entity_dic:
        if list(set(entity_dic[file][entity_dic[file]["main entity"] == 1]['column']) & set(master_id)):
            main_tables.append(file)
    for file in entity_dic:
        if list(set(entity_dic[file][entity_dic[file]["main entity"] == 1]['column'])):
            if file not in main_tables:
                main_tables.append(file)
    return main_tables


def find_forward_backward(data_tables, entity_dic):
    """Find relation between table"""
    lst_file = list(data_tables.keys())
    df = pd.DataFrame(columns=lst_file, index=lst_file)
    df.index.name = 'file'
    df = df.reset_index()
    # to read by feature from  left to top
    for file in data_tables:
        for file_compare in data_tables:
            if file != file_compare:
                intersec_cols = list(data_tables[file].columns & data_tables[file_compare].columns)
                if ((find_main_id(entity_dic[file])) in intersec_cols) or (
                        (find_main_id(entity_dic[file_compare])) in intersec_cols):
                    if (find_main_id(entity_dic[file])) and (find_main_id(entity_dic[file_compare])):
                        if (find_main_id(entity_dic[file])) in intersec_cols:
                            df.loc[df['file'] == file_compare, file] = "FW"
                        if (find_main_id(entity_dic[file_compare])) in intersec_cols:
                            df.loc[df['file'] == file_compare, file] = "BW"
                    elif find_main_id(entity_dic[file]):
                        df.loc[df['file'] == file_compare, file] = "FW"
                    elif find_main_id(entity_dic[file_compare]):
                        df.loc[df['file'] == file_compare, file] = "BW"
                    else:
                        df.loc[df['file'] == file_compare, file] = "M2M"
            else:
                df.loc[df['file'] == file_compare, file] = "NO"
    return df
