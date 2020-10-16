import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
names = ["tilt","left_weight","left_distance","right_weight", "right_distance"]
dataset = pandas.read_csv(url, names=names)

# scatter_matrix(dataset)
# plt.show()

array = dataset.values

X = array[:,1:5]
Y = array[:,0]

validation_size = 0.20
seed = 10

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

print(X_train,"\n")
print(X_validation,"\n")
print(Y_train,"\n")
print(Y_validation,"\n")

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
best = ["__",None,0]

for name, model in models:
    kfold = model_selection.KFold(n_splits=25, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    if (best[2] < cv_results.mean()):
        best[0] = name
        best[1] = model
        best[2] = cv_results.mean()

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print(cv_results)

print("\n")
# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

best[1].fit(X_train, Y_train)
predictions = best[1].predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))