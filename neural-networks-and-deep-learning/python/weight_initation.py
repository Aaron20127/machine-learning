"""weight_initialization 
~~~~~~~~~~~~~~~~~~~~~~~~
This program shows how weight initialization affects training.  In
particular, we'll plot out how the classification accuracies improve
using either large starting weights, whose standard deviation is 1, or
the default starting weights, whose standard deviation is 1 over the
square root of the number of input neurons.
"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

def main(filename, n, eta):
    run_network(filename, n, eta)
    make_plot(filename)
                       
def run_network(filename, n, eta):
    """Train the network using both the default and the large starting
    weights.  Store the results in the file with name ``filename``,
    where they can later be used by ``make_plots``.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)

    print "Train the network using the default starting weights."
    default_vc, default_va, default_tc, default_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,            
        # = net.SGD(training_data[:11], 30, 11, eta, lmbda=5.0,
                  evaluation_data=validation_data,
                #   monitor_training_cost = True, 
                  monitor_evaluation_accuracy=True)

    print "Train the network using the large starting weights."
    net.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,        
        # = net.SGD(training_data[:11], 30, 11, eta, lmbda=5.0,        
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)

    print "Train the network using the default starting weights 1."
    net.default_weight_initializer_1()
    default_vc_1, default_va_2, default_tc_3, default_ta_4 \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,        
        # = net.SGD(training_data[:11], 30, 11, eta, lmbda=5.0,        
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)


    f = open(filename, "w")
    json.dump({"default_weight_initialization":
               [default_vc, default_va, default_tc, default_ta],
               "large_weight_initialization":
               [large_vc, large_va, large_tc, large_ta], 
                "default_weight_initialization_1":
               [default_vc_1, default_va_2, default_tc_3, default_ta_4]},
              f)
    f.close()

def make_plot(filename):
    """Load the results from the file ``filename``, and generate the
    corresponding plot.
    """
    f = open(filename, "r")
    results = json.load(f)
    f.close()

    default_vc, default_va, default_tc, default_ta = results[
        "default_weight_initialization"]

    large_vc, large_va, large_tc, large_ta = results[
        "large_weight_initialization"]
    # Convert raw classification numbers to percentages, for plotting

    default_vc_1, default_va_1, default_tc_1, default_ta_1 = results[
        "default_weight_initialization_1"]

    default_va = [x/100.0 for x in default_va]

    large_va = [x/100.0 for x in large_va]

    default_va_1 = [x/100.0 for x in default_va_1]    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.arange(0, 30, 1), large_va, color='#9932CC', linewidth=2.0,
            label="Old approach to weight initialization")

    ax.plot(np.arange(0, 30, 1), default_va, color='#FF4040', linewidth=2.0,
            label="New approach to weight initialization")

    ax.plot(np.arange(0, 30, 1), default_va_1, color='#FFA933', linewidth=2.0,
            label="New approach to weight initialization 1")

    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main("haha", 30, 0.1)
