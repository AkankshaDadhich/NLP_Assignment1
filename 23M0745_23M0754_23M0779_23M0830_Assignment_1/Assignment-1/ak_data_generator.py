import numpy as np
import random
import csv
from tqdm import tqdm  # Import tqdm for progress bar



class NeuralNetwork:


    
    def __init__(self, no_of_inputs, no_of_hidden_neurons, no_of_outputs , momentum=0.5):
        self.no_of_inputs = no_of_inputs 
        self.no_of_hidden_neurons = no_of_hidden_neurons 
        self.no_of_outputs = no_of_outputs 
        self.momentum = momentum  # Momentum parameter



        # Initialize weights randomly
        self.weights_input_hidden = 2*np.random.rand(self.no_of_inputs, self.no_of_hidden_neurons) -1  # +1 for bias
        # print(self.weights_input_hidden)
        self.weights_hidden_output = 2*np.random.rand(self.no_of_hidden_neurons, self.no_of_outputs) - 1  # +1 for bias
        # print(self.weights_hidden_output)
        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        # Forward propagation through the network
        self.hidden_layer_activation = self.sigmoid(np.dot(X, self.weights_input_hidden))
        self.output = self.sigmoid(np.dot(self.hidden_layer_activation, self.weights_hidden_output))
        # print(self.output.shape)

    def backward_propagation(self, X, Y, learning_rate):
        # Backpropagation
        error = Y - self.output
        d_output = error * self.sigmoid_derivative(self.output)
        # print(d_output.shape)
        # print(self.weights_hidden_output.T.shape)
        error_hidden_layer = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_activation)

        # Update weights

        # # Momentum
        # self.weights_hidden_output += self.hidden_layer_activation.T.dot(d_output) * learning_rate + self.momentum * self.weights_hidden_output
        # self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate + self.momentum * self.weights_input_hidden
        
        # Without Momentum
        self.weights_hidden_output += np.dot(self.hidden_layer_activation.T, d_output) * learning_rate
        self.weights_input_hidden += np.dot(X.T, d_hidden_layer) * learning_rate

    def train(self, X, Y, learning_rate):
        self.forward_propagation(X)
        self.backward_propagation(X, Y, learning_rate)

    def predict(self, X):
        self.forward_propagation(X)
        return self.output

    def get_accuracy(self, X, Y):
        correct_predictions = 0

        predictions = self.predict(X)
        for i in range(len(predictions)):
            if predictions[i] >= 0.5 and Y[i] == 1:
                correct_predictions += 1
            elif predictions[i] < 0.5 and Y[i] == 0:
                correct_predictions += 1

        accuracy = correct_predictions / len(Y)
        return accuracy

    def get_precision(self, X, Y):
        true_positives = 0
        false_positives = 0

        predictions = self.predict(X)
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:
                if Y[i] == 1:
                    true_positives += 1
                else:
                    false_positives += 1

        if true_positives + false_positives == 0:
            return 0

        precision = true_positives / (true_positives + false_positives)
        return precision

    def get_confusion_matrix(self, X, Y):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        predictions = self.predict(X)
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:
                if Y[i] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if Y[i] == 0:
                    true_negatives += 1
                else:
                    false_negatives += 1

        return [[true_positives, false_positives],
                [false_negatives, true_negatives]]
    





def number_to_bits(number, number_of_bits):
    return np.array([int(i) for i in np.binary_repr(number, width=number_of_bits)]).tolist()


def is_palindrome(number):
    return 1 if number == number[::-1] else 0

def data_generate(max_number , number_of_bits):
    data = []
    for i in range(max_number):
        input_data = number_to_bits(i, number_of_bits)
        #input_data.extend([1])
        output = is_palindrome(input_data)
        input_data.append( output)
        if output==1 :
            for i in range(31):
                data.append(input_data)
        data.append(input_data)
    random.shuffle(data)
    return data

def main():
    data = []
    max_number = 1023
    csv_file = 'Assignment-1\\data1.csv'
    number_of_bits = 10
    data = data_generate(max_number, number_of_bits)
    epochs = 3000
    learning_rate = 0.005
    #print(data)  
    write_to_csv(csv_file, data)
    X, Y = read_from_csv(csv_file)    
    n = len(X)
    # print(n )
    # for i in range(10):
    #     print(X[i] , "X")
    #     print( Y[i] , "Y")
    # return
    XX = [ X[:n//4], X[n//4:2*n//4], X[2*n//4:3*n//4], X[3*n//4:] ]
    YY = [ Y[:n//4], Y[n//4:2*n//4], Y[2*n//4:3*n//4], Y[3*n//4:] ]

    for i in range(4):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for j in range(4):
            if i == j:
                X_test.extend(XX[j])
                Y_test.extend(YY[j])
            else:
                X_train.extend(XX[j])
                Y_train.extend(YY[j])
    

        neural_net = NeuralNetwork(no_of_inputs=10, no_of_hidden_neurons=10, no_of_outputs=1)
        #return
        for epoch in tqdm(range(epochs)):
            # Y=np.array(Y_train).reshape(-1, 1)
            # print(Y.shape)
            # X=np.array(X_train)
            # print(X.shape)
            neural_net.train(X=np.array(X_train), Y=np.array(Y_train).reshape(-1,1), learning_rate=learning_rate)
            
            # Check progress after every 100 epochs
            if (epoch + 1) % 100 == 0:
                progress = ((epoch + 1) / epochs) * 100
               # print(f'Epoch {epoch + 1}/{epochs} - Progress: {progress:.1f}%')
        
        # print('Hidden Layer Weights:')
        # print(neural_net.weights_input_hidden)
        # print('Output Layer Weights:')
        # print(neural_net.weights_hidden_output)

        print('Epoch:', epoch + 1)
        print('Accuracy = ', neural_net.get_accuracy(X=np.array(X_test), Y=np.array(Y_test)))
        print('Precision = ', neural_net.get_precision(X=np.array(X_test), Y=np.array(Y_test)))
        print('Confusion Matrix:', neural_net.get_confusion_matrix(X=np.array(X_test), Y=np.array(Y_test)))



 

def write_to_csv(csv_file, data):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data   )

def read_from_csv(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        #print(data)
    X = [list(map(int, row[:-1])) for row in data]
    Y = [(int(row[-1])) for row in data]
    #print ("X", X , "y\n", Y )
    
    return X , Y







if __name__ == "__main__":
    main()


