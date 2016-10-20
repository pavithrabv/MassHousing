# Third-party libraries
import numpy as np
def convertfunc(x):
    y =float(x)/100000000;
    return y



# Load Training data and test data from CSV files
def load_data_from_csv():
    raw_training_csv_file = '../data/training_data_batch1.csv'
    raw_test_csv_file = '../data/test_data_batch1.csv'

    raw_training_data = np.genfromtxt(raw_training_csv_file, delimiter=',', dtype='f8',
                                      filling_values=0.00000000, skip_header=1,
                                      usecols=(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                                      converters={0: convertfunc, 2: convertfunc, 3: convertfunc,
                                                  4: convertfunc, 5: convertfunc, 6: convertfunc,
                                                  7: convertfunc, 8: convertfunc,9: convertfunc,
                                                  10: convertfunc, 11: convertfunc, 12: convertfunc,
                                                  13: convertfunc})
    raw_test_data = np.genfromtxt(raw_test_csv_file, delimiter=',', dtype='f8', filling_values=0.00000000,
                                  skip_header=1, usecols=(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                                  converters={0: convertfunc, 2: convertfunc, 3: convertfunc,
                                              4: convertfunc, 5: convertfunc, 6: convertfunc,
                                              7: convertfunc, 8: convertfunc, 9: convertfunc,
                                              10: convertfunc, 11: convertfunc, 12: convertfunc,
                                              13: convertfunc})

    return raw_training_data, raw_test_data


def get_data_for_analysis():
    raw_training_data, raw_test_data = load_data_from_csv()

    training_input_tuples = tuple(x[0:len(x)-1] for x in raw_training_data)
    training_inputs = [np.reshape(x, (-1, 1)) for x in training_input_tuples]
    #test_print(training_inputs)

    training_grade_tuple = tuple(x[len(x)-1] for x in raw_training_data)
    training_results = [vectorize(y) for y in training_grade_tuple]

    training_data = zip(training_inputs, training_results)
    #test_print(training_data)

    test_input_tuples = tuple(x[0:len(x)-1] for x in raw_test_data)
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
