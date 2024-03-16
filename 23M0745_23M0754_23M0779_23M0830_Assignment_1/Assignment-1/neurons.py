import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = 2 * np.random.rand(input_size) - 1

    def find_output(self, input):
        return np.dot(self.weights, input)
    
    def outer_layer_get_delta_j(self, y_actual, y_pred):
        delta_j = (y_actual - y_pred) * y_pred * (1 - y_pred)
        return delta_j
    
    def hidden_layer_get_delta(self, output, next_weight, next_delta_j):
        delta_j = next_weight * next_delta_j * output * (1 - output)
        return delta_j
    
    def weight_change(self, delta_j, previous_inputs, learning_rate):
        previous_inputs = np.array(previous_inputs)
        change_in_weight = learning_rate * delta_j * previous_inputs
        self.weights += change_in_weight
    
    def print_weights(self):
        print(self.weights)

    def get_weights(self):
        return self.weights