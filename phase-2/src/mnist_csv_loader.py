# Third-party libraries
import numpy as np
import math
from sklearn import preprocessing

#def convertfunc(x):
#    if not x:
 #       y = 0.0
  #  else:
   #     y = float(x)
    #y =float(x)
    #if(math.isnan(y)):
    #    y = 0.0
#  return y


# Load Training data and test data from CSV files
def load_data_from_csv():
    raw_training_csv_file = '../data/phase2_trainingdata.csv'
    raw_test_csv_file = '../data/phase2_testdata.csv'

    raw_training_data = np.genfromtxt(raw_training_csv_file, delimiter=',', dtype='float',filling_values=0, skip_header=1)
                                      # ,usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    raw_test_data = np.genfromtxt(raw_test_csv_file, delimiter=',', dtype='float', filling_values=0, skip_header=1)
   # usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))

    return raw_training_data, raw_test_data


def get_data_for_analysis():
    raw_training_data, raw_test_data = load_data_from_csv()

    normalized_training_data = preprocessing.normalize(raw_training_data, norm='l2')
    normalized_test_data = preprocessing.normalize(raw_test_data, norm='l2')

    training_input_tuples = tuple(x[1:len(x)] for x in normalized_training_data)
    training_inputs = [np.reshape(x, (-1, 1)) for x in training_input_tuples]
    #test_print(training_inputs)

    training_grade_tuple = tuple(x[0] for x in raw_training_data)
    training_results = [vectorize(y) for y in training_grade_tuple]

    training_data = zip(training_inputs, training_results)
    #test_print(training_data)

    test_input_tuples = tuple(x[0:len(x)-1] for x in normalized_test_data)
    test_inputs = [np.reshape(x, (-1, 1)) for x in test_input_tuples]

    test_grade_tuple = tuple(x[len(x)-1] for x in raw_test_data)
    test_data = zip(test_inputs, test_grade_tuple)

    return training_data, test_data


def vectorize(z):
    e = np.zeros((5, 1))
    e[int(z) - 1] = 1.0
    return e


def test_print(training_inputs):
    for row in training_inputs:
        print (row)