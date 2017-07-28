import mnist_loader
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
import network22
net = network22.Network([784, 100, 10], cost=network22.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)