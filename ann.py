import numpy as np

# FUNKCJE POMOCNICZE
#1. funkcja aktywacji
# wzór g(z)=1/(1+e^-z)
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-1 * z))
    return g
#2.
def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# inicjalizacja
class ANN(object):

    def __init__(self, nodes_in_hidden_layer=10):
        self._nodes_in_hidden_layer = nodes_in_hidden_layer
        self._weights_matrix_1 = None
        self._weights_matrix_2 = None


    # funkcja przepuszczajca dane przez NN
    def _NN_feed_forward(self, input_data):
        #pobranie liczby probek uczacych i rozmiarow macierzy input_data (m x 784)
        num_of_ip_samples, input_data_dim = input_data.shape
        #dodanie kolumny jedynek (bias unit) do macierzy input_data (=> m x 785)
        ones_column = np.ones((num_of_ip_samples, 1))
        ip_with_bias = np.hstack((ones_column, input_data))
        #mnozenie macierzy z macierza wag (785 x 20); z_1 - aktywatory hidden layer
        z_1 = np.dot(ip_with_bias, self._weights_matrix_1.T)
        #przepuszczenie aktywatorow przez f.simgmoidalna => output w (0,1)
        a_1 = sigmoid(z_1)
        #dodanie kolumny jedynek (bias unit) do a_1
        a_1_with_bias = np.hstack((ones_column, a_1))
        #mnozenie a_1_with_bias z macierza wag; z_2 - aktywatory output layer
        z_2 = np.dot(a_1_with_bias, self._weights_matrix_2.T)
        #przepuszczenie aktywatorow przez f.simgmoidalna => output w (0,1)
        a_2 = sigmoid(z_2)
        #macierz przewidywanych wartosci wyjsciowych
        return a_2

