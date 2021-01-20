import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import tree
from sklearn import svm


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

iris = pd.read_csv('./Data/iris.csv')

def show_data():
    return iris

def knn():
    x = iris.iloc[:, 0:4]
    y = iris.iloc[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    n_neighbors = int(math.sqrt(len(y)))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='euclidean')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred)

def logistic_regression():
    x = iris.iloc[:, 0:4]
    y = iris.iloc[:, 4]
    x = np.array(x)
    y = np.array(y)

    model = LogisticRegression(solver='liblinear', random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def decision_tree_classifier():
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    Decision_Tree_Entropy = DecisionTreeClassifier(criterion="entropy", random_state=100)
    Decision_Tree_Entropy.fit(x_train, y_train)
    y_pred = Decision_Tree_Entropy.predict(x_test)
    return accuracy_score(y_test, y_pred)


def random_forest():
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    clf = rfc(n_jobs=2)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)


def support_vector_machines():
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    classifier = svm.SVC(kernel='linear', random_state=100)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return accuracy_score(y_test, y_pred)


def get_function_names():
    names = ["knn", "rf", "lr", "dtc", "svm"]
    return names

def get_function_full_name(function_name):
    
    classifiers = {
        "knn": "K Nearest Neighbours",
        "lr": "Logistic Regression",
        "dtc": "Decision Tree Classifier",
        "rf": "Random Forest",
        "svm": "Support Vector Machines"
    }

    return classifiers[function_name]

def get_function_acronym(function_name):
    
    classifiers = {
        "K Nearest Neighbours":"knn",
        "Logistic Regression":"lr",
        "Decision Tree Classifier":"dtc",
        "Random Forest":"rf",
        "Support Vector Machines":"svm"
    }

    return classifiers[function_name]
