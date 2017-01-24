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


