import numpy as np

class MatureNetwork:
    def __init__(self) -> None:
        
        self.configuration = self.loadConfig()
        self.layout = self.configuration[0]
        self.weights = self.configuration[1]
        self.biases = self.configuration[2]


    def guessNumber(self, flat_input_array):
        
        #Make input as first activation
        activations = []
        activations.append(flat_input_array)

        #Feed forward
        for l in range(0, len(self.layout) - 1):
            #compute weighted sum
            z = np.add(np.dot(self.weights[l], activations[l]), self.biases[l])

            '''
            print(self.weights[l].shape)
            print(activations[l].shape)
            print(z.shape)
            '''
            
            #Get activation
            a = self.sigmoid(z)
            activations.append(a)

        #Get activation for last layer        
        guessed_digit = 0
        final_vector = activations[len(activations) - 1]

        #Get index for highest activation value
        for a in range(0, len(final_vector)):
            if(final_vector[a] > final_vector[guessed_digit]):
                guessed_digit = a

        return guessed_digit
    


    def loadConfig(self):

        with open('NetworkConfig.npy','rb') as file:

            setup = np.load(file, allow_pickle=True) #Get network layout
            weights = []
            biases = []

            #For each layer 
            for i in range(0, len(setup) - 1):
                m = np.load(file, allow_pickle=True)
                weights.append(m)


            for i in range(0, len(setup) - 1):
                m = np.load(file, allow_pickle=True)
                biases.append(m)

        return [setup, weights, biases]
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))