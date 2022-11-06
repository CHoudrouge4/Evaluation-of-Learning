import numpy as np
import statistics

# Compute Sensitivity
def sensitivity(TP, Pos):
    return TP/Pos

# Compute Specificity
def specificity(TN, Neg):
    return TN/Neg

# Compute false positive rate
def false_pos_rate(FP, Neg):
    return FP/Neg

# compute false negative rate
def false_neg_rate(FN, Pos):
    return FN/Pos

# compute precision
def precision(TP, FP):
    return TP/(TP + FP)

# compute the recalls and precisions
def compute_recalls_precisions(cm):
    print(cm)
    TP = cm[0, 0]
    print(cm[0, 1])
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    Pos = TP + FN
    Neg = FP + TN
    return [sensitivity(TP, Pos), specificity(TN, Neg), false_pos_rate(FP, Neg), false_neg_rate(FN, Pos), precision(TP, FP)]

# create a tex file for the stat labels of sensitivity, specificity ....
def display_stat_latex_table(l):
    tex = '\\begin{table}[!h]\n'
    tex += '\\begin{tabular}{l | l | l| l| l | l}\n'
    tex += 'Model & Sensitivity & Specificity & False Positive Rate & False Negative Rate & Precision \\\\\\hline\n'
    models = ['DT', 'RF', 'SVM', 'KNN']
    for i in range(4):
        tex +=  models[i] + ' & ' + str(round(l[i][0], 4)) + ' & ' + str(round(l[i][1], 4)) + ' & ' + str(round(l[i][2], 4)) + ' & ' + str(round(l[i][3], 4)) + ' & ' + str(round(l[i][4], 4)) + '\\\\\n'
    tex += '\\end{tabular}\n'
    tex += '\\caption{}\n'
    tex += '\\end{table}\n'
    return tex

# convert a confusion matrix to latex representation
def cm_to_latex(cm):
    tex = '\\begin{tabular}{l | l | l| l}\n'
    tex +='                   & Predict $\oplus$ & Predict $\circleddash$ & \\\\\\hline\n'
    tex +='Actual $\oplus$ &' + str(cm[0][0]) + '&' + str(cm[0][1]) + '&' + str(cm[0][0] + cm[0][1]) + '\\\\ \n'
    tex +='Actual $\circleddash$ &' + str(cm[1][0]) + '&' + str(cm[1][1]) + '&' + str(cm[1][0] + cm[1][1])+ '\\\\\hline \n'
    tex +=                      '&' + str(cm[0][0] + cm[1][0]) + '&'+ str(cm[0][1] + cm[1][1]) +  '&' +  str(cm[0][0] + cm[0][1] +  cm[1][0] + cm[1][1]) + '\n'
    tex +='\end{tabular}\n'
    return tex

# it takes a list of cms
# create a table for the four confusion matrices
def display_cm_latex_code(l):
    tex = '\\begin{table}[!h]\n'
    names = ['DT', 'RF', 'SVM', 'KNN']
    for i in range(4):
        tex += '\\subfloat[' + names[i] + ']{\n'
        tex += cm_to_latex(l[i])
        tex += '}\n'
        tex += '\\hfill\n'

    tex += '\\caption{}\n'
    tex += '\\end{table}'
    return tex

# writing the tables of confusion matrices to file
def get_tex_file(l, i):
    tex = display_cm_latex_code(l)
    with open('./results/table' + str(i)+ '.tex', 'w') as f:
        f.write(tex)

# Write the precision and recalls table to a file
def get_stat_tex(l, i):
    tex = display_stat_latex_table(l)
    with open('./results/stat_table' + str(i)+ '.tex', 'w') as f:
        f.write(tex)

def cross_validation_table(stats):
    tex = '\\begin{table}[!h]\n'
    tex += '\\begin{tabular}{l| l  l  l  l  l  l}\n'
    tex += 'Model & DT & RF & SVM & KNN & MLP & GBC \\\\\\hline\n'
    for j in range(10):
        tex += str(j)
        for i in range(len(stats)):
            tex += '&' + str(round(stats[i][j], 4))
        tex += '\\\\\n'
    tex += '\\\\\\hline\n'
    tex += 'mean'
    for l in stats:
        tex += '&' + str(round(statistics.mean(l), 4))
    tex += 'stddev'
    for k in stats:
        tex += '&' + str(round(statistics.stdev(k), 4))
    tex += '\\\\\\hline\n'
    tex += '\\end{tabular}\n'
    tex += '\\caption{}\n'
    tex += '\\end{table}\n'
    return tex

def get_cross_validation_table(stats):
    tex = cross_validation_table(stats)
    with open('./results/cross_val' + '.tex', 'w') as f:
        f.write(tex)
