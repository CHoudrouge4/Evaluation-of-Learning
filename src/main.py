import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from preprocessing import *
from models import *
from calculations import *

def draw_roc(X_result, y_test, name):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, X_result)
    #print(fpr)
    #print(tpr)
    #roc_auc = metrics.auc(fpr, tpr)
    auc = round(metrics.roc_auc_score(y_test, X_result), 5)
    plt.plot(fpr, tpr,label= name + ", AUC="+str(auc))

# Function to produce the roc curve based on proba
def draw_roc1(X_test, y_test, proba, name):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, proba)
    #print(fpr)
    #print(tpr)
    auc = round(metrics.roc_auc_score(y_test, proba), 4)
    plt.plot(fpr, tpr,label= name + ", AUC="+str(auc))


#X stands for the features, labes for the class in question and i is for the drug number
# the function perform training and classication for a test and training data.
def experiment(X, labels, i):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=1)
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    X_train_fs, X_test_fs = feature_selection(X_train, X_test, y_train_enc, y_test_enc)
    #X_result = model_DT(X_train_fs, X_test_fs, y_train_enc, y_test_enc)

    models = [model_DT, model_RF, model_SVM, model_KNN]
    l = []
    stats = []
    for m in models:
        (X_result, X_proba), name = m(X_train_fs, X_test_fs, y_train_enc, y_test_enc)
        m_confusion = confusion_matrix(y_test_enc, X_result)
        #disp = ConfusionMatrixDisplay(confusion_matrix=m_confusion, display_labels={'nonuser', 'user'})
        #disp.plot()
        l.append(m_confusion)
        stats.append(compute_recalls_precisions(m_confusion))
        print(confusion_matrix(y_test_enc, X_result))
        #draw_roc(X_result, y_test_enc, name)
        draw_roc1(X_test_fs, y_test_enc, X_proba, name)
    get_tex_file(l, i)
    get_stat_tex(stats, i)
    plt.legend()
    plt.show()

# perform the experiments for each drug.
def main():
    file_name = './data/drug_consumption.data'
    print("Getting Data ...")
    data = get_data(file_name)
    #print(data.shape)
    classes = [31, 28, 24, 20, 16, 29]
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
