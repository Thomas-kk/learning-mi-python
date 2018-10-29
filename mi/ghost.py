# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

# Load dataset
names = [
    'bone_length', 'rotting_flesh', 'hair_length',
    'has_soul', 'color', 'type']
names2 = [
    'bone_length', 'rotting_flesh', 'hair_length',
    'has_soul', 'color']
dataset = pandas.read_csv("data/train.csv", names=names)
real = pandas.read_csv("data/test.csv", names=names2)
real.index.name = "id"

for column in dataset.columns:
    if dataset[column].dtype == type(object) and not column == "type":
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])

for column in real.columns:
    if real[column].dtype == type(object) and not column == "type":
        le = LabelEncoder()
        real[column] = le.fit_transform(real[column])

# Split-out validation dataset
array = dataset.values
size = 5
X = array[:, 0:size]  # Split the array into on that has the values
Y = array[:, size]  # and one that has the label
validation_size = 0.20
seed = 4
X_train, X_validation,
Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 4
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(max_depth=500)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Make predictions on validation dataset
knn = GaussianNB()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

pre = knn.predict(real)
# print(pre)
submissions = pandas.DataFrame(pre, index=real.index, columns=["type"])
submissions.to_csv("data/submission.csv", index=True)
