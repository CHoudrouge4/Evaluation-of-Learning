from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Train and test a model.
def model_exec(clf, X_train_fs, X_test_fs, y_train, y_test):
    clf = clf.fit(X_train_fs, y_train) #fitting the training data
    X_result = clf.predict(X_test_fs)  # testing
    X_proba = clf.predict_proba(X_test_fs)
    #print(accuracy_score(y_test, X_result))
    return X_result, X_proba[:,1]

# train an test Decision tree.
def model_DT (X_train_fs, X_test_fs, y_train, y_test):
    clf = tree.DecisionTreeClassifier() #creating the model
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test), 'DT'

# train and test random forest
def model_RF(X_train_fs, X_test_fs, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test), 'RF'

# train and test support vector machine
def model_SVM(X_train_fs, X_test_fs, y_train, y_test):
    clf = svm.SVC()
    clf=svm.SVC(probability=True)
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test), 'SVM'

# train and test k nearest neighbors
def model_KNN (X_train_fs, X_test_fs, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test), 'KNN'
