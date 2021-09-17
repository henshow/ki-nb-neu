# This is a sample Python script.
from matplotlib import pyplot as plt
from numpy import array, dot, zeros, random
import matplotlib.pyplot
from random import choice
import sklearn
import pandas
#import tensorflow
# import keras

"""
STRG+SHIFT+I Definition
SHIFT+ALT+UP or DOWN Zeile verschieben
ALT+UP or DOWN
STRG+ALT+LEFT or RIGHT Navigation der Positionen

"""
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

heaviside = lambda x: 0 if x < 0 else 1

def fit(iterations, training_data_set, w):
    errors = []
    weights = []
    for i in range(iterations):
        training_data = choice(training_data_set)
        x = training_data[0]
        y = training_data[1]
        y_hat = heaviside(dot(w, x))
        error = y - y_hat
        errors.append(error)
        weights.append(w)
        w += error * x
    return errors, weights



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.

def main():
    training_data_set = [
        (array([1, 0, 0]), 0),
        (array([1, 0, 1]), 1),
        (array([1, 1, 0]), 1),
        (array([1, 1, 1]), 1),
    ]

    random.seed( 12 )
    w = zeros(3)
    iterations = 30
    errors, weights = fit(iterations, training_data_set, w)
    w = weights[iterations - 1]
    print("Gewichtsfaktor am Ende des Training:")
    print(w)

    print("Auswertung am Ende des Trainings")
    for x, y in training_data_set:
        y_hat = heaviside(dot(x, w))
        print("{}: {} -> {}".format(x, y, y_hat))

    fignr = 1
    plt.figure(fignr, figsize=(10,10))
    plt.plot(errors)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Iteration')
    plt.ylabel(r"$(y - \hat y)$")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
