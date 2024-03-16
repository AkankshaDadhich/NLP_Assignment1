from data_reader import get_data_from_csv
from neural_networks import NeuralNetwork
from heat_map_generator import show_heat_map

def demo(neural_net):
    while True:
        x = input("Enter a numbers: ")
        x = [int(char) for char in x]
        if len(x)!=10:
            print("Invalid input")
            continue
        x.insert(0, 1)
        if neural_net.predict_one(x):
            print("It is palindrome")
        else:
            print("It is not a palindrome")



if __name__ == '__main__':

    file_path = '/home/akanksha/Music/nlp assignment/23M0745_23M0754_23M0779_23M0830_Assignment_1/Assignment-1/data.csv'
    X, Y = get_data_from_csv(csv_file_path=file_path)

    n = len(X)
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
        
        neural_net = NeuralNetwork(no_of_inputs=10, no_of_hidden_neurons=7, no_of_outputs=1)
        neural_net.train(X=X_train, Y=Y_train, no_of_epoch=500, learning_rate=0.05)
        print('Accuracy = ',neural_net.get_accuracy(X=X_test, Y=Y_test))
        print('Precision = ', neural_net.get_precision(X=X_test, Y=Y_test))
        print('Confusion Matrix:', neural_net.get_confusion_matrix(X=X_test, Y=Y_test))

        neural_net.print_weights()
        # demo(neural_net)
        hidden_layer_weights = neural_net.get_hidden_layer_weights()
        show_heat_map(hidden_layer_weights)
        # break
