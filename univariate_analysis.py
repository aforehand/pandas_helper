import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize

# returns the number of values in a df row
def how_many(row):
    if type(row) == str:
        try:
            row = eval(row)
        except:
            row = row.split()
    if type(row) == list:
        return len(row)
    else:
        return 1

# returns a column of tokenized lowercase strings with puctuation and stopwords
# removed. stemmer can be 'porter', 'snowball' or 'lancaster'
def clean_strings(column, stemmer=None):
    stemmer_dict = {'porter': stem.porter.PorterStemmer(),
                    'snowball': stem.snowball.EnglishStemmer(),
                    'lanaster': stem.lancaster.LancasterStemmer()}
    stop_words = stopwords.words('english')
    column = column.str.replace('[^\w\s]','')
    column = column.apply(lambda x: str(x).lower())
    column = column.apply(lambda x: word_tokenize(x))
    column = column.apply(lambda x: [word for word in x if word not in stop_words])
    if not stemmer is None:
        column = column.apply(lambda x: [stemmer_dict[stemmer].stem(word) for word in x])
    return column

# returns a list of elements in a given row
# elements may be dicts or strings
def _format_row(row):
    if type(row) == str:
        try:
            row = eval(row)
            if not (type(row)==list or type(row)==dict):
                row = str(row).split()
        except:
            row = row.split()
    elif type(row) == dict:
        row = [row]
    return row

# several functions use the same logic so now that logic lives here
# checks the element type, makes sure keys and values of dicts are
# valid, executes given functions when appropriate
def _do_common_logic(func, row, key, value, k, v, freq_dict={}):
    row = _format_row(row)
    if func == _freq_dict_accum:
        accumulator = freq_dict
    else:
        accumulator = 0
    if not (row is None or len(row)==0):
        for element in row:
            if (type(element)==dict and
                    not key is None and
                    (k is None and v is None or element[k]==v) and
                    (value is None or element[key]==value)):
                accumulator = func(accumulator, element[key], freq_dict)
            elif (type(element)==str and
                    value is None or element==value):
                accumulator = func(accumulator, element, freq_dict)
    return accumulator

# returns a frequency dict for the values in a df column
def get_freq_dict(column, key=None, value=None, k=None, v=None):
    freq_dict = {}
    for row in column:
        freq_dict = _do_common_logic(_freq_dict_accum, row, key, value, k, v, freq_dict)
    return freq_dict

def _freq_dict_accum(freq_dict, key, old_freq_dict):
    # if type(freq_dict)==int:
    #     freq_dict = old_freq_dict
    if key in freq_dict.keys():
        freq_dict[key] += 1
    else:
        freq_dict[key] = 1
    return freq_dict

# returns the column frequency of values in a df row
def get_freq_score(row, freq_dict, key=None, value=None, k=None, v=None):
    freq_score = 0
    freq_score = _do_common_logic(_freq_score_accum, row, key, value, k, v, freq_dict)
    return freq_score

def _freq_score_accum(freq_score, key, freq_dict):
    if key in freq_dict.keys():
        freq_score += freq_dict[key]
    return freq_score

# returns the number of occurrances of the given values in a given row
def get_value_count(row, key=None, value=None, k=None, v=None):
    count = 0
    count = _do_common_logic(_value_count_accum, row, key, value, k, v)
    return count

def _value_count_accum(*args):
    count = args[0]
    count += 1
    return count

# returns a column with the frequency of a value if that value is in
# each row or the sum of the frequencies of all values if no value
# is given.
# k and v are optional key/value pair that the frequency counting depends on
def get_freq_col(column, freq_dict=None, key=None, value=None, k=None, v=None):
    if freq_dict is None:
        freq_dict = get_freq_dict(column, key, value, k, v)

    freq_col = column.apply(lambda x: get_freq_score(x, freq_dict, key, value, k, v))
    return freq_col

# returns a column with values extracted from dicts in another column
def get_value_col(column, key='name', k=None, v=None):
    column = column.apply(lambda x: _format_row(x))

    def get_value_row(row, key, k, v):
        values = []
        if len(row) > 0 and not type(row[0])==dict:
            return row
        else:
            for d in row:
                if (k is None and v is None) or d[k]==v:
                    if key in d.keys():
                        values.append(d[key])
            return values

    value_col = column.apply(lambda x: get_value_row(x, key, k, v))
    return value_col

# returns a sorted dataframe of value frequencies
def get_freq_df(freq_dict=None, column=None, key=None, k=None, v=None, stemmer=None):
    if not column is None:
        column = clean_strings(column, stemmer)
    if freq_dict is None:
        freq_dict = get_freq_dict(column, key, k, v)

    df =  pd.DataFrame({'value': [k for k in freq_dict.keys()], 'frequency': [ v for v in freq_dict.values()]})
    # df = df[~df.value.isin(stopwords.words('english'))]
    df = df[df.value!='']
    df.sort_values(by='frequency', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#distribution plots for numeric data
def dist_plots(df, bins=None, kde=True):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for i in range(len(num_cols)):
        sns.distplot(train[num_cols[i]])
        plt.figtext(0.7,.8,f'skew: {train[num_cols[i]].skew():.2f}', size='medium')
        plt.figtext(0.7,.75,f'kurt: {train[num_cols[i]].kurt():.2f}', size='medium')
        plt.show()

#bar plots for categorical data
def bar_plots(df):
    cat_cols = df.select_dtypes(include=[np.object]).columns.tolist()
    # fig, axes = plt.subplots(len(cat_cols),1, figsize=(5,5*len(cat_cols)))
    for i in range(len(cat_cols)):
        sns.barplot(df[cat_cols[i]], ax=axes[i])
        plt.show()

# returns the z score of an element in a given column
