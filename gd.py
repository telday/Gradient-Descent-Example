#!/usr/bin/python3
"""
    file: gd
    author: Ellis Wright
    description: This file Reads in absorbancy data from my chemistry class
    and fits a linear regression model to the data using the single input
    feature expanded to polynomial form
"""
import numpy as np
from matplotlib import pyplot as plt

def load_data(filename):
    '''
    loads the data in from a file and returns the x values (wavelength of
    light) and the y values (absorbancy) as a tuple

    parameters:
        filename: the name of the file to read from

    return:
        A tuple with the first value being a list of the wavelengths
        and the second a list of absorbancy values
    '''
    f = open(filename, 'r')
    x = []
    y = []
    for i in f.readlines():
        # The file contians one datapoint on each line: wavelength absorbance
        x += [float(i.split()[0])]
        y += [float(i.split()[1])]
    f.close()
    return (x, y)

def power(x, power):
    '''
    Raises each value in an array (x) to the power (power)

    parameters:
        x: the array to raise to a power
        power: the power to raise the array to
    '''
    for i in range(len(x)):
        x[i] = x[i]**power

def create_training_set(x):
    '''
    Creates the training set by adding an additional feature to the set for
    each new power function. This is done so that a set with only one feature
    can seem like it has more than one.

    Ex. If the original feature was x then the new feature array would be
    x x^2 x^3 x^4

    parameters:
        x: the initial array of values to expand into a featureset matrix

    return:
        Returns the new matrix as a numpy.array
    '''
    new_array = []
    for i in range(1, 4):
        power(x, i)
        # Equivalent of copy.deepcopy(x)
        new_array += [x[:]]
    return np.asarray(new_array)

def normalize_data(x):
    '''
    Puts the data into a workable format (all numbers close to 0). This is
    done by subtracting each value in the feature by the mean of that 
    featureset and dividing each value by the standard deviation for
    that featureset.

    parameters:
        x: The numpy.array object to normalize

    return:
        A normalized version of the array
    '''
    new_x = x
    #This function uses lists to ensure that an array with any number 
    #of features can be normalized

    #List of averages for each featureset
    mean = []
    #List of std deviations for each featureset
    sigma = []
    
    #For each column in the given array
    for i in range(x.shape[1]):
        #new_x[:, i] gets all values in column i of new_x as a numpy.array obj
        mean += [np.mean(new_x[:, i])]
        sigma += [np.std(new_x[:, i])]

    #For each row in the given array
    for i in range(x.shape[0]):
        #Subtract the mean and divide by the std deviation for each value
        new_x[i, :] = np.subtract(new_x[i, :], mean)
        new_x[i, :] = np.divide(new_x[i, :], sigma)

    #Mean and sigma will be used to normalize any data the user would 
    #want to apply the output of the algorithm too
    return (new_x, mean, sigma)


def gradient_descent(training_set, expected, theta, alpha):
    '''
    Performs one update in the gradient descent algorithm:
    theta(j) = theta(j) - (alpha / m) * sum((X*THETA - y)X(j))

    Where alpha is the learning rate, m is the number of training sets
    X is the training set matrix, THETA is the perameter matrix and X(j)
    is featureset j of the perameter matrix.

    parameters:
        training_set: The normalized set of training data
        expected: The expected output of each training set
        theta: The current theta value
        alpha: The learning rate

    returns:
        The new value of theta after one iteration of gradient descent
    '''
    # X*THETA - y
    delta = np.subtract(training_set.dot(theta), expected)
    # X^T * (X*THETA - y)
    delta = training_set.transpose().dot(delta)
    # THETA = THETA - alpha/m * (X^T * (X*THETA - y))
    new_theta = np.subtract(theta, alpha * (1/training_set.shape[0]) * delta)
    return new_theta

def cost_function(training_set, expected, theta):
    '''
    Returns the value of the cost function with the given perameters.
    J(theta) = (1/2m) * sum( (X*THETA - y).^2 )

    parameters:
        training_set: The normalized set of training data
        expected: The expected output of each training set
        theta: The current theta value
    
    return:
        The value of the cost function applied with these parameters
    '''
    # X*THETA
    delta = training_set.dot(theta)
    # X*THETA - y
    delta = np.subtract(delta, expected)
    # (X*THETA - y).^2
    square = np.multiply(delta, delta)
    # sum((X*THETA - y).^2)
    summ = np.sum(square)
    # (1/2m) * sum((X*THETA - y).^2)
    return (1.0/(2.0 * training_set.shape[0])) * summ
    
def main():
    # Load the data and initialize the training_set and expected values
    init_x, init_y = load_data("Absorbance_Data.dat")
    x = create_training_set(np.asarray(init_x)).transpose()
    y = np.asarray(init_y)
    
    # We only need to normalize the input here so our output still makes sense
    x, mean, sigma = normalize_data(x)
    
    #Insert a column of ones to x so the THETA(0) value is not lost
    one = np.ones(x.shape[0]).tolist()
    x = x.transpose().tolist()
    x.insert(0, one)
    x = np.asarray(x).transpose()
    
    # Set the initial learning rate, theta value and first cost value
    alpha = 0.01
    theta = np.zeros(x.shape[1])
    cost_history = [cost_function(x, y, theta)]

    # Perform the gradient descent algorithm 100000 times. You could also
    # Compare the last two values in cost history for an acceptable error.
    for i in range(100000):
        theta = gradient_descent(x, y, theta, alpha)
        cost_history += [cost_function(x, y, theta)]
    print(theta)

    # Plot the cost history as a function of iterations
    plt.plot(range(len(cost_history)), cost_history)
    plt.show()

    # We have to scale the original input values so they fit the normalized
    # data. 
    initial_input_scaled = x[:, 1]
    expected_output = y
    actual_output = x.dot(theta)

    # Plot the actual values, and the linear regression model
    plt.scatter(initial_input_scaled, expected_output)
    plt.plot(initial_input_scaled, actual_output)
    plt.show()

if __name__ == "__main__":
    main()
