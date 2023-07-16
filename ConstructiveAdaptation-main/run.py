import numpy as np
import pandas as pd
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import argparse
EPS = np.finfo(float).eps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='Dataset to run.')
args = parser.parse_args()

# load dataset
if args.dataset.lower() == 'credit':
    exec(open("credit.py").read())
elif args.dataset.lower() == 'german':
    exec(open("german.py").read())
elif args.dataset.lower() == 'spam':
    exec(open("spam.py").read())
elif args.dataset.lower() == 'bank':
    exec(open("bank.py").read())
else:
    exec(open("adult.py").read())

# cost matrix
s_I, s_M = 1, 5
S = np.identity(N_I + N_M)
for i in range(N_I):
    S[i][i] = s_I
for i in range(N_M):
    S[i+N_I][i+N_I] = s_M

# load the baseline methods
exec(open("LightTouch.py").read())

clf = LogisticRegression(fit_intercept=True).fit(x, y)
print(f"No Startegic Behavior Train Accuracy: {(clf.predict(x)==y).mean()}")


# create a pd dataframe to store the results.
results_df = pd.DataFrame(columns=['test_accuracy', 'test_accuracy_std', 'accuracy', 'accuracy_std', 'fraction', 'fraction_std', 'deterioration', 'deterioration_std'])
skf = StratifiedKFold(n_splits=5)
accuracy = []
fraction = []
test_acc = []
deterioration = []
w_static = []

for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
    acc0, acc, frac, dete, w = run_svm(X_train, y_train, X_test, y_test)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
    test_acc.append(acc0)
    w_static.append(np.array(w))
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
w_static = np.array(w_static)

results_df.loc['Static'] = [test_acc.mean(), test_acc.std(), accuracy.mean(), accuracy.std(), fraction.mean(), fraction.std(), deterioration.mean(), deterioration.std()]

test_acc = []
accuracy = []
fraction = []
deterioration = []
w_drop_features = []
for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x_I[train], y[train], x_I[test], y[test]
    acc0, acc, frac, dete, w = run_improvable_svm(X_train, y_train, X_test, y_test)
    test_acc.append(acc0)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
    w_drop_features.append(np.array(w))
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
w_drop_features = np.array(w_drop_features)

results_df.loc['DropFeatures'] = [test_acc.mean(), test_acc.std(), accuracy.mean(), accuracy.std(), fraction.mean(), fraction.std(), deterioration.mean(), deterioration.std()]

test_acc = []
accuracy = []
fraction = []
deterioration = []
w_manipulation_proof = []
for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
    acc0, acc, frac, dete, w = run_strategic_clf(X_train, y_train, X_test, y_test)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
    test_acc.append(acc0)
    w_manipulation_proof.append(np.array(w))
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
w_manipulation_proof = np.array(w_manipulation_proof)

# print(f"${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")
results_df.loc['ManipulationProof'] = [test_acc.mean(), test_acc.std(), accuracy.mean(), accuracy.std(), fraction.mean(), fraction.std(), deterioration.mean(), deterioration.std()]
# lbds = np.arange(0., 1.5, 0.1)
lbds = [1.]
for l in lbds:
    accuracy = []
    test_acc = []
    fraction = []
    deterioration = []
    w_lighttouch = []
    for train, test in skf.split(x, y):
        X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
        acc0, acc, frac, dete, w = run_recourse_clf(X_train, y_train, X_test, y_test, lbd=l)
        test_acc.append(acc0)
        accuracy.append(acc)
        fraction.append(frac)
        deterioration.append(dete)
        w_lighttouch.append(np.array(w))
    test_acc = np.array(test_acc)
    accuracy = np.array(accuracy)
    fraction = np.array(fraction)
    deterioration = np.array(deterioration)
    w_lighttouch = np.array(w_lighttouch)

results_df.loc['LightTouch'] = [test_acc.mean(), test_acc.std(), accuracy.mean(), accuracy.std(), fraction.mean(), fraction.std(), deterioration.mean(), deterioration.std()]

# print the results in a 2 tables, one for the metrices and one for the stds
print(results_df.iloc[:, ::2])
print(results_df.iloc[:, 1::2])

# Notes
print("Notes:")
print("1. test_accuracy: accuracy of the classifier (without strategic behavior) on the test set")
print("2. accuracy: (%) of examples classified correctly even after manipulation")
print("3. fraction: (%) changes from 0 (true label) to 1 (predicted) after improvement")
print("4. deterioration: (%) changes from 1 (true label) to 0 (predicted) after improvement")

# plot the weights mean +- std all in one plot
plt.figure(figsize=(10, 5))
plt.errorbar(np.arange(1, 1+len(w_static[0])), w_static.mean(axis=0), yerr=w_static.std(axis=0), label='Static')
plt.errorbar(np.arange(1, 1+len(w_drop_features[0])), w_drop_features.mean(axis=0), yerr=w_drop_features.std(axis=0), label='DropFeatures')
plt.errorbar(np.arange(1, 1+len(w_manipulation_proof[0])), w_manipulation_proof.mean(axis=0), yerr=w_manipulation_proof.std(axis=0), label='ManipulationProof')
plt.errorbar(np.arange(1, 1+len(w_lighttouch[0])), w_lighttouch.mean(axis=0), yerr=w_lighttouch.std(axis=0), label='LightTouch')

# set xticks to 'columns'
plt.xticks(np.arange(1, 1+len(w_static[0])), [c[:len(c)//2] + "\n" + c[len(c)//2:] for c in columns] + ["bias"], rotation=90)

# create a vertical line at x=x.shape[1]+1 and write "total" on it
plt.axvline(x=x.shape[1]+1, color='k', linestyle='--')
plt.text(x=x.shape[1]+1.1, y=0, s='total', rotation=90, verticalalignment='bottom')


# create a vertical line at N_I and write "improvement" on it
plt.axvline(x=N_I, color='k', linestyle='--')
plt.text(x=(0 + N_I) / 2, y=0, s='improvement', verticalalignment='center')

# create a vertical line at N_I+N_M and write "manipulation" on it
plt.axvline(x=N_I+N_M, color='k', linestyle='--')
plt.text(x=(N_I + N_I+N_M) / 2, y=0, s='manipulation', verticalalignment='center')

# create a vertical line at (N_I+N_M + x.shape[1]+1) / 2 and write "non-actionable" on it
plt.text(x=(N_I+N_M + x.shape[1]+1) / 2, y=0, s='non-actionable', verticalalignment='center')

# add pad to the bottom of the plot
plt.subplots_adjust(bottom=0.3)


plt.legend()
plt.savefig(f"{args.dataset}_weights.png")
plt.show()

