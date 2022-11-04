import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

from preprocessing import *
from models import *
from calculations import *


# implement cross validation
def cross_validation(X, y, model, mode):
# TODO use oversamoling/undersampling, give the option as param
    #print(X)
    oversample = RandomOverSampler(sampling_strategy='minority')
    undersample = NearMiss(version=1, n_neighbors=3)
    kfold = KFold(n_splits=10)
    for train_index, test_index in kfold.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # implement oversampling
        if mode == 1:
            X_train, y_train = oversample.fit_resample(X, y)
        if mode == 2:
            X_train, y_train = undersample.fit_resample(X, y)
        (y_pred, X_proba), name = model(X_train, X_test, y_train, y_test)
        print(name)
        print(accuracy_score(y_test, y_pred))
    return 0


#X stands for the features, labes for the class in question and i is for the drug number
# the function perform training and classication for a test and training data.
def experiment(X, labels, i):
# TODO: Normalize the Data
# TODO: use the kflod things
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    y_enc = prepare_targets(labels)
    X_fs = feature_selection(X_scaled, labels)
    #X_result = model_DT(X_train_fs, X_test_fs, y_train_enc, y_test_enc)

    models = [model_GBC]
    #model_DT, model_RF, model_SVM, model_KNN,
    l = []
    stats = []
    for m in models:
        cross_validation(X_fs, y_enc, m, 1)

# perform the experiments for each drug.
def main():
    file_name = './data/drug_consumption.data'
    print("Getting Data ...")
    data = get_data(file_name)
    #print(data.shape)
    #classes = [31, 28, 24, 20, 16, 29]
    classes = [31]
    for c in classes:
        convert_to_binary_class(data, c)

    X = data[:, 1:13]    # getting the features
    i = 1
    for c in classes:
        labels = data[:, c] #getting class label
        experiment(X, labels, i)
        i = i + 1

if __name__ == "__main__":
    main()
