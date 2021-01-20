import machine_learning_functions as milf
from matplotlib import pyplot as plt

def get_trends():
    dataset = milf.show_data()

    # Show Charts
    sepallength = dataset.iloc[:, 0]
    sepalwidth = dataset.iloc[:, 1]
    petallength = dataset.iloc[:, 2]
    petalwidth = dataset.iloc[:, 3]

    plt.plot(petallength, color="green", linewidth = 1, label='Petal Length')
    plt.plot(petalwidth, color="blue", linewidth = 1, label='Petal Width')
    plt.plot(sepalwidth, color="orange", linewidth = 1, label='Sepal Length')
    plt.plot(sepallength, color="red", linewidth = 1, label='Sepal Width')
    
    plt.title("Trend of Features")
    plt.grid(True)
    return plt