import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# This is my markdown Area, I can Write anythin here
Anything you write here gets converted into a Web Page
""")

# Adding Side Bar
st.sidebar.header('Header Bar')

def user_input_features():
    option1 = st.sidebar.slider('Option 1', 4.3, 7.9, 5.4)
    option2 = st.sidebar.slider('Option 2', 2.0, 4.4, 3.4)
    option3 = st.sidebar.slider('Option 3', 1.0, 6.9, 1.3)
    option4 = st.sidebar.slider('Option 4', 0.1, 2.5, 0.2)
    data = {'option1': option1,
            'option2': option2,
            'option3': option3,
            'option4': option4}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Add Subheader
st.subheader("Iris Dataset")

# Print a Dataframe or a table - Updates with the value change in the side bar
iris = pd.read_csv('./Data/iris.csv')
st.write(iris)

# Drawing Line Chart
petallength = iris.iloc[:, 2]
st.write(""" 

### Trend in Petal length
""")
st.line_chart(petallength)

# Drawing Bar Chart
st.write("""

### Histogram or Bar Chart
""")

st.bar_chart(petallength)

# Using matplotlib
from matplotlib import pyplot as plt

# part 1
x = np.arange(1, 300, 5)
y = np.arange(1, 61)
y2 = np.arange(1, 2 * 60, 2)
plt.plot(x, y, color="green", linewidth = 4, linestyle = ':')
plt.plot(x, y2, color="blue", linewidth = 2)
plt.title("Basic Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)

st.pyplot(plt)

# part 2
plt.scatter(x, y)
plt.title("Basic Plot 2")
plt.xlabel("X axis")
plt.ylabel("Y axis")
st.pyplot(plt)

# part 3

df = pd.DataFrame(
np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
columns=['lat', 'lon'])
st.map(df)

# Buttons
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')

# Sliders
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

from datetime import time, datetime
st_time = st.slider(
    "When do you start?",
    value=(datetime(2020, 1, 1, 9, 30), datetime(2020, 12, 12, 9, 30)),
    format="MM/DD/YY - hh:mm")
st.write("Start time:", st_time)

# taking input
title = st.text_input('Movie title', '')
st.write(title)

para = st.text_area('Add a long paragraph here: ', '')
st.write(para)

time_lunch = st.time_input('Input time for lunch:')
st.write(time_lunch)

st.file_uploader('Upload file here')
color = st.color_picker('Pick Colour Here')
st.write(f'colour chosen is {color}')

col1, col2, col3 = st.beta_columns(3)
with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)
with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)
with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)

col1, col2 = st.beta_columns([3, 1])
data = np.random.randn(10, 1)
col1.subheader("A wide column with a chart")
col1.line_chart(data)
col2.subheader("A narrow column with the data")
col2.write(data)

st.line_chart({"data": [1, 5, 2, 6, 2, 1]})
with st.beta_expander("See explanation"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

with st.echo():
   st.write('This code will be printed')

bar = st.progress(50)
