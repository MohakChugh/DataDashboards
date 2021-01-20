import streamlit as st
import pandas as pd
import numpy as np
import machine_learning_functions as milf

st.write(""" 
# Dashboard for IRIS Dataset Prediction
A Web UI for IRIS dataset. Calculate accuracy scores
and predictions for KNN, Logistic regression, SVM and more
""")

show_dataset = st.button('Click here to see the dataset')

if show_dataset:
    st.subheader("Iris Dataset")
    st.write(milf.show_data())
    if st.button('Close Dataset Window'):
        show_dataset = not show_dataset

st.sidebar.header('Select Classifier Type')

classifiers = {
    "K Nearest Neighbours":"knn",
    "Logistic Regression":"lr",
    "Decision Tree Classifier":"dtc",
    "Random Forest":"rf",
    "Support Vector Machines":"svm"
}

# Managing Names and Functions
function_acronyms = milf.get_function_names()
function_names = []
for i in range(len(function_acronyms)):
    function_names.append(milf.get_function_full_name(function_acronyms[i]))

functions = {
    "knn": milf.knn,
    "lr": milf.logistic_regression,
    "dtc": milf.decision_tree_classifier,
    "rf": milf.random_forest,
    "svm": milf.support_vector_machines,
    "show_data": milf.show_data
}

select_classifier = st.sidebar.selectbox(
    "Select Classifier",(function_names)
)

selected_function = functions[milf.get_function_acronym(select_classifier)]

if selected_function:
    st.write(f'Currently selected Algorithm is ***{select_classifier}***')
    st.write(f'The Current Accuracy Score for the Algorithm is: {selected_function() * 100}%')
