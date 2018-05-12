"""
This is reposible for dumping txt file for xlearn API
"""
import pandas as pd


__all__ = ['dump_libffm_file']


def dump_libffm_file(data, label, f='data_ffm.txt',
                     feature_name='auto',
                     categorical_feature='auto'):
    """Dump the dataset in libffm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    NOTE: some format comparisons:
    libsvm format:

        label index_1:value_1 index_2:value_2 ... index_n:value_n

    CSV format:

        value_1 value_2 .. value_n label

    libffm format:

        label field_1:index_1:value_1 field_2:index_2:value_2 ...

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : {array-like, sparse matrix}, shape = [n_samples (, n_labels)]
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : string, specifies the path where the libffm data is saved.

    feature_name : Feature names.
        list of strings or 'auto', optional (default="auto"))
        If ‘auto’ and data is pandas DataFrame, data columns names are used.

    categorical_feature : Categorical features.
        list of strings or 'auto', optional (default="auto"))
        If list of strings, interpreted as feature names (need to specify feature_name as well).   # noqa
        If ‘auto’ and data is pandas DataFrame, pandas categorical columns are used.

    References
    ----------
    Large Scale and Sparse Data Learning - FAFM using xlearn
    https://www.analyticsvidhya.com/blog/2018/01/factorization-machines/
    """
    # handle data type
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # handle feature list setup
    if feature_name == 'auto':
        feature_name = list(data.columns)
    if categorical_feature == 'auto':
        categorical_feature = \
            list(data.select_dtypes(include=['category']).columns)
    numeric_feature = [f for f in feature_name if f not in categorical_feature]
    currentcode = len(numeric_feature)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numeric_feature:
        catdict[x] = 0
    for x in categorical_feature:
        catdict[x] = 1

    nrows = data.shape[0]
    with open(str(f), "w") as text_file:
        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = data.iloc[r].to_dict()
            datastring += str(label[n])
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x] == 0):
                    datastring = '{0} {1}:{2}:{3}'.format(datastring, str(i), str(i), str(datarow[x]))    # noqa
                else:
                    # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    code = catcodes[x][datarow[x]]
                    datastring = '{0} {1}:{2}:1'.format(datastring, str(i), str(int(code)))    # noqa
            datastring += '\n'
            text_file.write(datastring)
