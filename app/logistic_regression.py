import numpy as np
import pandas as pd


def read_csv(file_path):
    """ Citește un fișier CSV și returnează un DataFrame Pandas. """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Eroare la citirea fișierului CSV: {e}")
        return None


def prepare_data(df):
    """ Separăm simptomele (X) de etichete (y). """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    max_values = np.max(X, axis=0)
    max_values[max_values == 0] = 1
    X_normalized = X / max_values
    return X_normalized, y


def sigmoid(z):
    """ Funcția sigmoidă pentru regresia logistică. """
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """ Funcția softmax pentru probabilități multiclase. """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """ Algoritmul de gradient descent pentru optimizarea parametrilor theta. """
    m = len(y)
    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        theta -= learning_rate * gradient
    return theta


def train_logistic_regression(X, y, num_classes, learning_rate=0.05, num_iterations=1000):
    """ Antrenarea modelului de regresie logistică multiclasa. """
    m, n = X.shape
    all_theta = np.zeros((num_classes, n))
    for i in range(num_classes):
        y_i = (y == i).astype(int)
        theta = np.zeros(n)
        all_theta[i, :] = gradient_descent(X, y_i, theta, learning_rate, num_iterations)
    return all_theta


def predict(all_theta, X):
    """ Calculăm probabilitățile pentru fiecare clasă și returnăm cele mai probabile clase. """
    return softmax(X @ all_theta.T)
