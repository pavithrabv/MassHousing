# Third-party libraries
#import numpy as np


import network2
import random
import mnist_csv_loader
training_data, test_data = mnist_csv_loader.get_data_for_analysis()


# create a list to represent 3-layer network
#  13 neurons in the input layer, corresponding to each data field per training example on the csv
#  14 hidden neurons (random choice, may be changed)
#  5 output neurons, corresponding to each of the 5 grade values
#middleLayerCount = random.randint(100,110)
#print "Middle Layer count {0}".format(middleLayerCount)
list = [51, 7, 5]
#
# # create a Network object
#     # initilize object with a list (see above)
#     # list specifies network structure
net = network2.Network(list)
#
# # begin training your Network object
#
#     # training takes place at:
#     #     30 complete epochs
#     #     999 training examples at one time (mini_batch_size)
#     #     0.1 learning rate
#     #     method params - SGD(self, training_data, epochs, mini_batch_size, eta, evaluation_data=None,
#     #                           monitor_training_accuracy=False):
#
#net.SGD(training_data=training_data, epochs=30, mini_batch_size=999, eta=0.1)
#
net.SGD(training_data=training_data, epochs=30, mini_batch_size=999, eta=0.1, lmbda=1.0, evaluation_data=test_data,
     monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    monitor_training_cost=True, monitor_training_accuracy=True)
