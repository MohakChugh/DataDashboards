import pandas as pd
import numpy as np
import math

# Classifiers for the model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import tree
from sklearn import svm

# metric Analysis - Currently Not required
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

iris = pd.read_csv('./Data/iris.csv')

def knn_pred(input):
    x = iris.iloc[:, 0:4]
    y = iris.iloc[:, 4]
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)

    n_neighbors = int(math.sqrt(len(y)))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='euclidean')
    classifier.fit(x, y)

    y_pred = classifier.predict(input)
    return y_pred

def logistic_regression_pred(input):
    x = iris.iloc[:, 0:4]
    y = iris.iloc[:, 4]
    x = np.array(x)
    y = np.array(y)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model = model.fit(x, y)

    y_pred = model.predict(input)
    return y_pred


def decision_tree_classifier_pred(input):
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]

    Decision_Tree_Entropy = DecisionTreeClassifier(criterion="entropy", random_state=100)
    Decision_Tree_Entropy.fit(x, y)

    y_pred = Decision_Tree_Entropy.predict(input)
    return y_pred


def random_forest_pred(input):
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]

    clf = rfc(n_jobs=2)
    clf.fit(x, y)

    y_pred = clf.predict(input)
    return y_pred


def support_vector_machines_pred(input):
    x = iris.iloc[:, :4]
    y = iris.iloc[:, 4]

    classifier = svm.SVC(kernel='linear', random_state=100)
    classifier.fit(x, y)
    y_pred = classifier.predict(input)
    return y_pred

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

def get_prediction_function(function_acronym):
    
    classifiers = {
        "knn": knn_pred,
        "lr": logistic_regression_pred,
        "dtc": decision_tree_classifier_pred,
        "rf": random_forest_pred,
        "svm": support_vector_machines_pred
    }

    return classifiers[function_acronym]

def get_function_acronym(function_name):
    
    classifiers = {
        "K Nearest Neighbours":"knn",
        "Logistic Regression":"lr",
        "Decision Tree Classifier":"dtc",
        "Random Forest":"rf",
        "Support Vector Machines":"svm"
    }

    return classifiers[function_name]