import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#-Functions ------------------------------------------------------------------#
def feature_normalise(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, ddof=1, axis=0)
    X = stats.zscore(X, ddof=1, axis=0)
    
    return X, mu, sigma


def compute_cost(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    J = 1 / (2 * m) * np.sum(np.square(h-y))
    
    return J


def gradient_descent(X, y, theta, alpha=0.01, n_iters=500):

    m = len(y)
    J_history = np.zeros((n_iters, 1))
    theta_history = np.zeros((n_iters, len(theta)))
    
    for i in range(n_iters):
        h = np.dot(X, theta)
        error = h - y
        theta -= ((alpha / m) * np.dot(X.T, error))
        theta_history[i, :] = theta.T
        J_history[i] = compute_cost(X, y, theta)
    
    return theta, J_history, theta_history


def predict(x, mu, sigma, theta):
    if type(x) == list:
        x = [1] + x # Add an intercept
    else:
        x = [1] + [x] # Turn into a list and add intercept
    print(x)
    xnorm = np.concatenate((np.array([x[0]]), np.array((x[1:] - mu) / sigma)))
    return np.dot(xnorm, theta)


#-Run the example ------------------------------------------------------------#
# Read the data using numpy
data = np.genfromtxt('ex1data2.txt', delimiter=',')
# Define which is the matrix of features `X`, and the labels vector `y`
X, y = data[:, 0:2], data[:, -1] # Change this according to data dimensions.
# Reshape y
y = y.reshape(len(y), 1)
# Normalise the features & get descriptives
X, mu, sigma = feature_normalise(X)
# Define some variables for the gradient descent function
alpha = 0.01
n_iters = 500
# Add an intercept of ones to the matrix of features
m = len(y) # get n cases
X = np.c_[np.ones((m, 1)), X]
# Calculate how many features
n_features = np.size(X, 1)
# Initialize theta
theta = np.zeros((n_features, 1))
# Run the grafient descent
theta, J_history, theta_history = gradient_descent(X, y, theta)
# Make a plot
plt.plot(J_history)
plt.xlabel('Iteration Number')
plt.ylabel('Jθ')
plt.show()
# Predict y for X1 = 1650; X2 = 3 given theta  
y_pred = predict(x=[1650, 3], mu=mu, sigma=sigma, theta=theta)
