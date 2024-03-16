from tqdm import tqdm
from neurons import Neuron
import numpy as np

class NeuralNetwork:
    def __init__(self, no_of_inputs, no_of_hidden_neurons, no_of_outputs):
        self.no_of_inputs = no_of_inputs
        self.no_of_hidden_neurons = no_of_hidden_neurons
        self.no_of_outputs = no_of_outputs

        self.hidden_neurons = [Neuron(no_of_inputs+1) for _ in range(no_of_hidden_neurons)]
        self.output_neurons = [Neuron(no_of_hidden_neurons+1) for _ in range(no_of_outputs)]

    def get_all_neuron_outputs(self, x):
        hidden_layer_outputs = [1]
        for neuron in self.hidden_neurons:
            output = neuron.find_output(x)
            output = self.sigmoid(output)
            hidden_layer_outputs.append(output)
        hidden_layer_outputs = np.array(hidden_layer_outputs)
        final_outputs = []
        for neuron in self.output_neurons:
            output = neuron.find_output(hidden_layer_outputs)
            output = self.sigmoid(output)
            final_outputs.append(output)
        final_outputs = np.array(final_outputs)
        outputs = {
            'hidden_layer_outputs' : hidden_layer_outputs,
            'final_outputs' : final_outputs
        }

        return outputs
    
    def predict(self, X):
        y_pred = [self.predict_one(x) for x in X]
        return y_pred

    def predict_one(self, x):
        if len(x) != self.no_of_inputs + 1:
            return -1
        outputs = self.get_all_neuron_outputs(x)
        return 1 if outputs['final_outputs'][0]>=0.5 else 0
    
    def cross_entropy_loss(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)
        loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        mean_loss = np.mean(loss)
        return mean_loss

    def train(self, X, Y, no_of_epoch, learning_rate):

        for _ in tqdm(range(no_of_epoch)):
            y_pred = []
            for x,y in zip(X,Y):
                x = np.array(x)
                outputs = self.get_all_neuron_outputs(x)

                hidden_layer_outputs = outputs['hidden_layer_outputs']
                final_outputs = outputs['final_outputs']
                y_pred.append(final_outputs[0])
                
                output_neuron_delta_j = self.output_neurons[0].outer_layer_get_delta_j(y, final_outputs[0])
                self.output_neurons[0].weight_change(output_neuron_delta_j, hidden_layer_outputs, learning_rate)

                for i, hidden_neuron in enumerate(self.hidden_neurons):
                    current_delta_j = hidden_neuron.hidden_layer_get_delta(hidden_layer_outputs[i+1], self.output_neurons[0].weights[i+1], output_neuron_delta_j)
                    hidden_neuron.weight_change(current_delta_j, x, learning_rate)

            # print(self.cross_entropy_loss(Y, y_pred))
                    
    def get_confusion_matrix(self, X, Y):
        conf_matrix = [[0, 0], [0, 0]]

        true_labels = Y
        predicted_labels = self.predict(X)
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            conf_matrix[true_label][predicted_label] += 1

        confusion_matrix = {
            'true_positive'     : conf_matrix[1][1],
            'false_positive'    : conf_matrix[0][1],
            'false_negative'    : conf_matrix[1][0],
            'true_negative'     : conf_matrix[0][0]
        }
        return confusion_matrix

    def get_accuracy(self, X, Y):
        confusion_matrix = self.get_confusion_matrix(X, Y)
        true_positive = confusion_matrix['true_positive']
        true_negative = confusion_matrix['true_negative']
        accuracy = (true_positive + true_negative) / len(X) 
        return accuracy
    
    def get_precision(self, X, Y):
        true_positive = 0
        false_positive = 0
        
        for x, y in zip(X, Y):
            y_pred = self.predict_one(x)
            if y_pred == 1:
                if y == 1:
                    true_positive += 1
                else:
                    false_positive += 1

        precision = true_positive / (true_positive + false_positive)
        return precision
    
    def get_hidden_layer_weights(self):
        hidden_layer_weights = []
        for neuron in self.hidden_neurons:
            hidden_layer_weights.append(neuron.get_weights())
        return hidden_layer_weights
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def manual_test(self):
        while True:
            num_integers = 10
            user_input = input(f"Enter {num_integers} digits of 0's and 1's without spaces: ")
            integer_list = [1] + [int(digit) for digit in user_input]
            print(self.predict_one(integer_list))

    def print_weights(self):
        for neuron in self.output_neurons:
            neuron.print_weights()
        print()
        for neuron in self.hidden_neurons:
            neuron.print_weights()
