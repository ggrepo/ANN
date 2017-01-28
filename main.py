import struct
import numpy as np
from ann import ANN
import matplotlib.pyplot as plt
from pylab import cm
import time
from datetime import timedelta

DATA_FOLDER = 'data'
TRAIN_IMGS_FILE = 'train-images.idx3-ubyte'
TRAIN_LBLS_FILE = 'train-labels.idx1-ubyte'
TEST_IMGS_FILE = 't10k-images.idx3-ubyte'
TEST_LBLS_FILE = 't10k-labels.idx1-ubyte'


def read_data():
    train_img_filepath = DATA_FOLDER + '/' + TRAIN_IMGS_FILE
    train_lbl_filepath = DATA_FOLDER + '/' + TRAIN_LBLS_FILE
    test_img_filepath = DATA_FOLDER + '/' + TEST_IMGS_FILE
    test_lbl_filepath = DATA_FOLDER + '/' + TEST_LBLS_FILE

    with open(train_lbl_filepath, 'rb') as tr_lbl_file:
        #pierwsze 4 bajty to magic number, nastepne 4 bajty to liczba etykiet
        train_lbl_magic_num, num_of_train_labels = struct.unpack(">2i", tr_lbl_file.read(8)) #2 intigery
        train_labels = np.fromfile(tr_lbl_file, dtype=np.uint8)

    with open(train_img_filepath, 'rb') as tr_img_file:
        #pierwsze 4 bajty to magic number, nastepne 4 bajty to liczba obrazkow,
        #kolejne 4 bajty to liczba wierszy w obrazku, kolejne 4 bajty to liczba kolumn w obrazku
        train_img_magic_num, num_of_train_images, tr_num_of_rows, tr_num_of_cols = struct.unpack(">4i", tr_img_file.read(16)) #4 intigery
        train_images = np.fromfile(tr_img_file, dtype=np.uint8).reshape(len(train_labels), tr_num_of_rows, tr_num_of_cols)

    with open(test_lbl_filepath, 'rb') as test_lbl_file:
        test_lbl_magic_num, num_of_test_labels = struct.unpack(">2i", test_lbl_file.read(8))
        test_labels = np.fromfile(test_lbl_file, dtype=np.uint8)

    with open(test_img_filepath, 'rb') as test_img_file:
        test_img_magic_num, num_of_test_images, tst_num_of_rows, tst_num_of_cols = struct.unpack(">4i", test_img_file.read(16))
        test_images = np.fromfile(test_img_file, dtype=np.uint8).reshape(len(test_labels), tst_num_of_rows, tst_num_of_cols)

    # print train_lbl_magic_num, num_of_train_labels, train_labels.shape
    # print train_img_magic_num, num_of_train_images, train_images.shape
    # print test_lbl_magic_num, num_of_test_labels, test_labels.shape
    # print test_img_magic_num, num_of_test_images, test_images.shape

    return train_labels, train_images, test_labels, test_images


raw_train_labels, raw_train_images, raw_test_labels, raw_test_images = read_data()

################

def reshape_input_data(ip_data):## zmiana obrazu mx(rzedy*kolumny)

	# wymiana danych wejsciowych
	num_of_samples, num_of_rows, num_of_cols = ip_data.shape
	return ip_data.reshape(num_of_samples, (num_of_rows*num_of_cols))


def reshape_output_data(op_data):

	#wymiana danych wyjsciowych
	num_of_samples = len(op_data)
	ret_data = np.zeros((num_of_samples, 10))
	for i in xrange(0, num_of_samples):
		index = op_data[i]
		ret_data[i,index] = 1
	return ret_data


# wyswietlanie kilku przykladowych obrazkow
# for i in range(3):
#     plt.imshow(raw_test_images[i], cmap=cm.gray)
#     plt.title("digit: {}".format(raw_test_labels[i]))
#     plt.show()


# modyfikacja obrazkow i etykiet za pomoca funkcji reshape
modif_train_images = reshape_input_data(raw_train_images)
modif_test_images = reshape_input_data(raw_test_images)
modif_train_labels = reshape_output_data(raw_train_labels)
modif_test_labels = reshape_output_data(raw_test_labels)
#print modif_test_images[0]
#print modif_test_labels[0]

#inicjalizacja ANN
ann = ANN(nodes_in_hidden_layer=20)

#trenowanie
start_time = time.time()
ann.train_NN(modif_train_images, modif_train_labels, num_of_iterations=10, learning_rate=0.1)
time_delta = time.time() - start_time
print "Training time: ", str(timedelta(seconds=time_delta))

#przewidywanie etykiet dla danych testowych
predicted_op = ann.estimate_op(modif_test_images)
#print predicted_op[0]
predicted_labels = ann.predict_NN(modif_test_images)
#print predicted_labels[0]

#obliczenie trafnosci ANN
total_labels = len(predicted_labels)
correct_labels = np.sum(predicted_labels == raw_test_labels)
accuracy = float(correct_labels)/total_labels * 100

#wyswietlenie wynikow
print "Total No. of Test Images: %d" % (total_labels)
print "Images Correct Classified: %d" % (correct_labels)
print "Accuracy of Model: %.2f%%" % (accuracy)

