import streamlit as st
import pandas as pd
import numpy as np
import machine_learning_functions as milf
import predictors as pred
import charts as charts

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

classifiers = {
    "K Nearest Neighbours":"knn",
    "Logistic Regression":"lr",
    "Decision Tree Classifier":"dtc",
    "Random Forest":"rf",
    "Support Vector Machines":"svm"
}

# Web-Page starts from here

st.write(""" 
# Dashboard for IRIS Dataset Prediction
A Web UI for IRIS dataset. Calculate accuracy scores
and predictions for KNN, Logistic regression, SVM and more
""")

show_dataset = st.button('Click here to see the dataset')

if show_dataset:
    st.subheader("Iris Dataset")
    st.write(milf.show_data())
    st.pyplot(charts.get_trends())

    if st.button('Close Dataset Window'):
        show_dataset = not show_dataset

st.sidebar.header('Select Classifier Type')

select_classifier = st.sidebar.selectbox(
    "Select Classifier",(function_names)
)

get_predictions = st.sidebar.checkbox('Get Predictions for this Classifier')

if get_predictions:
    sepallength = st.sidebar.slider('Sepal Length', 4.3, 7.9)
    sepalWidth = st.sidebar.slider('Sepal Width', 2.0, 4.4)
    petallength = st.sidebar.slider('Petal Length', 1.0, 6.9)
    petalwidth = st.sidebar.slider('Petal width', 0.1, 2.5)    
    
    st.write('The Current Values for the Prediction is: ')

    input_array = [sepallength, sepalWidth, petallength, petalwidth]
    input_array = np.array(input_array)
    input_array = input_array.reshape(1, -1)

    st.write(input_array)

    prediction_function = pred.get_prediction_function(pred.get_function_acronym(select_classifier))
    prediction = prediction_function(input_array)
    st.write(f'The Predicted Value is :', prediction)


selected_function = functions[milf.get_function_acronym(select_classifier)]

if selected_function and not get_predictions:
    st.write(f'Currently selected Algorithm is ***{select_classifier}***')
    st.write(f'The Current Accuracy Score for the Algorithm is: {selected_function()*100}%')
    st.write()

