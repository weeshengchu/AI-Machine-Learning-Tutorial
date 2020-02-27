import sklearn
from sklearn import datasets
# 1.8 Use svm as classifier
from sklearn import svm
# 1.9 Implementing SVM
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

# 1.8 Splitting Data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(x_train, y_train)

classes = ['malignant' 'benign']

# 1.9 Using SVC to train
# clf = svm.SVC()

# gives a better acc by adding in 'rbf' -> linear, poly, ... C=1 -> soft margin
# clf = svm.SVC(kernel="linear", C=2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)