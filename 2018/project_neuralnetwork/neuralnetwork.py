import numpy as np
# scipy.special for sigmoidfunksjonen expit()
import scipy.special
import matplotlib.pyplot


# en klasse for det nevrale nettverket
class neuralNetwork:
    
    #initaliserer det nevrale nettverket
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # bestemmer antall noder  i første, andre og tredje lag
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # kobler vekt-matrisene sammen, wih og who
        # vektene i matrisene er w_i_j, hvor koblinga er fra node i til node j i det neste laget
        # w11 w21
        # w12 w22 osv.
        # En alternativ måte å deklarere vektene på:
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # læringsrate
        self.lr = learningrate
        
        # activation function er sigmoidfunksjonen
        self.activation_function = lambda x: scipy.special.expit(x)

    # metode for å trene det nevrale nettverket
    def train(self, inputs_list, targets_list):
        # konverterer input lista til en todimensjonal matrise
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # kalkulerer input i lag 2
        hidden_inputs = np.dot(self.wih, inputs)
        # kalulerer output fra lag 2
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # kalkulerer input i det siste laget
        final_inputs = np.dot(self.who, hidden_outputs)
        # kalkulerer output fra det siste laget
        final_outputs = self.activation_function(final_inputs)
        
        # kalkulerer feilen i det siste laget (differansen mellom forventa verdi og verdien man får)
        output_errors = targets - final_outputs
        # feilen i det midterste laget er matriseproduktet av vektene mellom lag 2 og 3, og output_errors
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # se side 104 i MYONN
        # oppdaterer vektene mellom første lag og lag 2 
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

    # kobler noder og vekter sammen
    def query(self, inputs_list):
        # konverterer input til en todimensjonal array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # kalkulerer input i midterste lag
        hidden_inputs = np.dot(self.wih, inputs)
        # kalkulerer output fra midterste lag
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # kalkulerer input i siste lag
        final_inputs = np.dot(self.who, hidden_outputs)
        # kalkulerer output fra siste lag (outputen til det nevrale nettverket)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# antall input, hidden og output noder
input_nodes = 784 # 784 noder fordi input vil er bilde med 28x28 piksler
hidden_nodes = 100
output_nodes = 10

# læringsraten er 0.3
learning_rate = 0.3

# lager det nevrale nettverket
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# laster treningsdata fra MNIST i en liste
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# treningsdataen er gitt på følgende måte:
# første verdi er selve tallet (f.eks.: 5)
# resten av verdiene er piksel-verdier (0-255) som forteller hvordan selve tallet ser ut. Disse er separert med komma

# trener det nevrale nettverket
# går gjennom alle tall i treningsdataen
for record in training_data_list:
    # splitter verdiene med komma 
    all_values = record.split(',')
    # skalerer verdiene fra intervallet [0, 255] til [0.01, 0.99]
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # lager en liste med lengde 10 med verdier 0.01. Hver indeks tilsvarer et siffer 0-9.
    targets = np.zeros(output_nodes) + 0.01
    # setter den første verdien i lista til 0.99 fordi den verdien er selve sifferet
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)