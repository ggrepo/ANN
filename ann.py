# -*- coding: utf-8 -*-
import numpy as np

# FUNKCJE POMOCNICZE

#1. funkcja aktywacji - obliczana na podstawie wartosci wyjscia neuronow z sieci, uzywana  w sieciach jednokierunkowych
# wzór g(z)=1/(1+e^-z), input z: tablica a numpy
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-1 * z))
    # zwraca numpy array z elementacji "mądrej" funkcji sigmoidalnej kazdego elementu z
    return g
#2.
def sigmoidGradient(z):
    #input a  tablica numpy
    # zwraca numpy array  z elementami "uczącej się" funkcji sigmoidalnej-gradient na kazdym elemencie z
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# inicjalizacja
class ANN(object):

    def __init__(self, nodes_in_hidden_layer=20):
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
        #mnozenie a_1_with_bias z macierza wag; z_2 - aktywatory output layergit
        z_2 = np.dot(a_1_with_bias, self._weights_matrix_2.T)
        #przepuszczenie aktywatorow przez f.simgmoidalna => output w (0,1)
        a_2 = sigmoid(z_2)
        #macierz przewidywanych wartosci wyjsciowych
        return a_2



    # metoda pozwalająca na znalezienie ważenia   kosztów lub  błędów (najlepszych parametrów wagi H1 i H2)
    def _NN_compute_cost(self, output_data, estimated_output, reg_param):
            #pobieranie ilości próbek wyjsciowych i ich  wymiaru
            num_of_op_samples, num_of_labels = output_data.shape

            # użycie wektora do znalezienie kosztów
            temp_1 = (-1) * (output_data) * (np.log(estimated_output))
            temp_2 = (-1) * (1 - output_data) * (np.log(1 - estimated_output))
            cost = np.sum(temp_1) + np.sum(temp_2)
            norm_cost = cost/num_of_op_samples

            # dodanie współczynników ważenia  kosztów
            weights_1_no_bias = self._weights_matrix_1[:,1:]
            weights_2_no_bias = self._weights_matrix_2[:,1:]
            reg_cost = np.sum(np.square(weights_1_no_bias)) + np.sum(np.square(weights_2_no_bias))
            norm_reg_cost = reg_cost * (float(reg_param)/(2*num_of_op_samples))

            # koszt całkowity
            total_cost = norm_cost + norm_reg_cost
            return total_cost

    #należy dodać jeszcze metodę  dla pojedynczej iteracji!


    def _NN_backpropagation(self, output_data, input_data, estimated_output, reg_param):

        #pobranie liczby probek i rozmiarow macierzy output_data (m x 10)
        num_of_op_samples, num_of_labels = output_data.shape
        # pobranie liczby probek i rozmiarow macierzy input_data (m x 784)
        num_of_ip_samples, input_data_dim = input_data.shape

        #znajdujemy blad dla ostatniej warstwy
        delta_3 = estimated_output - output_data
        delta_2_half = np.dot(delta_3, self._weights_matrix_2)

        ones_column = np.ones((num_of_ip_samples, 1))
        ip_with_bias = np.hstack((ones_column, input_data))
        z_1 = np.dot(ip_with_bias, self._weights_matrix_1.T)
        a_1 = sigmoid(z_1)
        a_1_with_bias = np.hstack((ones_column, a_1))

        #propagujemy blad do hidden layer
        g_dash_z = sigmoidGradient(z_1)
        delta_2 = delta_2_half[:,1:] * g_dash_z
        #liczymy big_deltas
        big_delta_2 = np.dot(delta_3.T, a_1_with_bias)
        big_delta_1 = np.dot(delta_2.T, ip_with_bias)
        #liczymy gradienty
        grad_1 = big_delta_1/num_of_ip_samples
        grad_2 = big_delta_2/num_of_ip_samples
        #regularyzacja
        reg_weights_1 = (reg_param/num_of_ip_samples) * self._weights_matrix_1
        reg_weights_2 = (reg_param/num_of_ip_samples) * self._weights_matrix_2
        #zerujemy kolumne z bias
        reg_weights_1[:,0] = 0.0
        reg_weights_2[:,0] = 0.0
        #dodajemy regularyzacje do gradientow
        final_grad_1 = grad_1 + reg_weights_1
        final_grad_2 = grad_2 + reg_weights_2
        #gradienty dla macierzy wag
        return final_grad_1, final_grad_2

