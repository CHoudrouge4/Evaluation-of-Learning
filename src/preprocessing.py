import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, SelectFpr, GenericUnivariateSelect

# read data from file.
def get_data(file_name):
    # Read the file in pandas data frame
    data = pd.read_csv(file_name, header=None)
    # store the datasfrom sklearn.metrics import accuracy_scoreet
    dataset = data.values
    return dataset

# Convert CL0 and CL1 -> nonuser, the other five to users
def convert_to_binary_class(data, c):
    m, n = data.shape
    for i in range(m):
        if data[i][c] == 'CL0' or data[i][c] == 'CL1':
            data[i][c] = 'user'
        else:
            data[i][c] = 'nonuser'

# Binary Encoding
def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    #print(le.classes_)
    return y_enc

# Feature Selection
def feature_selection(X, y):
    fs = SelectKBest(score_func=f_classif, k=5)
    fs.fit(X, y)
    X_fs = fs.transform(X)

    for i in range(len(fs.scores_)):
    	print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    return X_fs
