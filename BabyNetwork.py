import numpy as np
import gzip
from typing import List


#Import the data into numpy array [Magic number, image num, row num, col num, rest]
#Flatten the data (764 rows)
#Normalize the data


#For each epochs
#For each training example
#Set 'a' = input
#Loop twice -> 
#calculate z = a*W + B
#append z to WSums array
#set a = sigmoid(z)

#If at the end of the loop:
#calculate deltaLast = dC/da*da/dz
#calculate deltaPrev = deltaLast * Transpose(W)* derivative(sigmoid(WSums[0]))
#Set W = W - LR*deltaPrev
#set B = B - LR*deltaPrev
#Calculate sumE += y - a / y, per training example
#calculate accuracy = sumE/nTraining
#Print(accuracy)


#function
#inputs: file path
#get file object from gzip
#read headers
#read pixel values
#re-arrange them into numpy array


#create feedworward function:
#First layer = input
#initialize weight matrix (W) with random values (maybe between 0 and 1) (764 x 128)
#initialize bias vector (B) with random values (764)
#Initialize WSums array
#Initialize LR = 0.000001
#Initialize sumError = 0
#Initialize epochs = 10


#dC/dw = a(l-1)*del
#dC/db = del 


#Note, rows are actually vectors i.e [1, 2, 3]

class BabyNetwork:


    #Biases are typically initalized to zero
    #Change weights initalization method, make them as small as possible else anything z = +/- 4, immidiately saturates the activation
    #Add ability to change input layer configuration
    #Make TrainNum and layers optional
    def __init__(self, TrainNum: int, layers: List[int]) -> None:

        self.weights = []
        self.biases = []
        self.layers = layers

        #Make weights according to the 'layers' array, Inputs will always be 764 length vector
        #width of the current matrix should be equal to the length of the previous to make the compatible
        # Initialize weights according to the 'layers' array
        for i in range(len(layers) - 1):
            layer_in = layers[i]  # Number of neurons in the current layer (input to the weights)
            layer_out = layers[i + 1]  # Number of neurons in the next layer (output from the weights)
            
            # Xavier initialization for weights
            variance = 2.0 / (layer_in + layer_out)
            weight = np.random.randn(layer_out, layer_in) * np.sqrt(variance)
            self.weights.append(weight)
            
            # Initialize biases for each layer to zero
            bias = np.zeros((layer_out, ))
            self.biases.append(bias)

        #Learning rate
        # Works for 600 examples and 30 neurons self.LR = 0.011111
        self.LR = 0.03
        
        self.TrainNum = TrainNum





    def testA(self,x):

        TrainingSet = self.getFormattedTraining('./MNIST dataset/t10k-images-idx3-ubyte.gz', './MNIST dataset/t10k-labels-idx1-ubyte.gz')

        for i in range(0,x):
            print('Set :' + str(TrainingSet[i][0]))
            print('weights: '+ str(self.weights[0]))
            print('Dot: ')
            print(str(np.dot(TrainingSet[i][0], self.weights[0])))

#TODO: Make the network reshuffle the training exapmes each epochs


    #Make compatible with minibatch
    #Note: order of dot proudct matters
    def Evaluate(self, epochs, batch_size):

        TrainingSet = self.getFormattedTraining('./MNIST dataset/train-images-idx3-ubyte.gz', './MNIST dataset/train-labels-idx1-ubyte.gz')

        n_batches = max(self.TrainNum // batch_size, 1)

        for e in range(0,epochs):

            np.random.shuffle(TrainingSet)
            ##Arrays containing output vectors for each training example

            guessed_right = 0

            prev_guessed = 0



            for batch_index in range(0, n_batches):

                weight_factors = [np.zeros_like(w) for w in self.weights]
                bias_factors = [np.zeros_like(b) for b in self.biases]

                start = batch_index *batch_size
                end = start +batch_size
                mini_batch = TrainingSet[start:end]

    

                for i in range(0,len(mini_batch)):

                    total_errors = []

                    activations = []

                    weighted_sums = []

                    #Append the image as the first activation for the image
                    #Shape: (1,784)
                    activations.append(mini_batch[i][0])

                    #For each layer

                    for l in range(0, len(self.layers) - 1):
                        #compute weighted sum
                        z = np.add(np.dot(self.weights[l], activations[l]), self.biases[l])

                        weighted_sums.append(z)
                        #Get activation
                        a = self.sigmoid(z)
                        activations.append(a)

                    #print('Activations: '+ str(activations[i]))
                    
                    guessed_digit = 0
                    final_vector = activations[len(activations) - 1]


                    guessed_digit = np.argmax(final_vector)

                    correct_digit = np.argmax(mini_batch[i][1])

                    if(guessed_digit == correct_digit):
                        guessed_right += 1



                    #Back propagation
                    #calculate error vector
                    #del(L) = (aL−y)⊙sigmoid(zL)

                    for l in range(0, len(self.layers) - 2):
                        total_errors.append([])

                        
                    #first get error for last layer using expected vector of current training image
                    #dL = np.multiply(np.subtract(activations[len(activations) - 1] , mini_batch[i][1]), self.sigmoid_prime(weighted_sums[len(weighted_sums) - 1])) 
                    #total_errors.append(dL)

                    dL = activations[len(activations) - 1] - mini_batch[i][1]
                    total_errors.append(dL)

                    #print('Error vector: '+str(dL))
                    #For each layer, compute their own error vectors
                    #Start from the second last layer
                    #del(l + 1) = the last appended vector in tota_errors
                    #Since first activation is always input image, to get the activation for the current layer you go one over
                    #((wl+1)Tδl+1)⊙sigmoid_prime(zl)
                    for l in range(0, len(self.layers) - 2)[::-1]:
                        dl = np.multiply(np.dot(np.transpose(self.weights[l + 1]), total_errors[l + 1]), self.sigmoid_prime(weighted_sums[l]))
                        total_errors[l] = dl


                    for l in range(0, len(self.layers) - 1):
                        weight_factors[l] += np.dot(total_errors[l].reshape(total_errors[l].size, 1), np.transpose(activations[l].reshape(activations[l].size, 1)))
                        bias_factors[l] +=  total_errors[l]
                
                

                self.weights = [w - (self.LR / batch_size) * dw for w, dw in zip(self.weights, weight_factors)]
                self.biases = [b - (self.LR / batch_size) * db for b, db in zip(self.biases, bias_factors)]



                '''
                if((prev_guessed - guessed_right) >= 100):
                    for l in range(0, len(self.layers) - 1):
                        weight_factors[l] = weight_factors[l] * -1
                        bias_factors[l] = bias_factors * -1
                '''
                #self.tune(weight_factors, bias_factors)
            

            print('Epoch: '+ str(e + 1) + ' accuracy: ' + str((guessed_right/self.TrainNum)))

            if(e == (epochs - 1)):
                self.saveConfig()
                #np.savez('SavedParameters',self.layers,self.weights,self.biases)
                print('Config, weights and biases have been saved to file.')
                




        
            
    def saveConfig(self):

        with open('NetworkConfig.npy', 'wb') as file:

            np.save(file, np.array(self.layers))

            for layer in self.weights:
                np.save(file, np.array(layer))

            for layer in self.biases:
                np.save(file, np.array(layer))

    '''
    #Adjust weights, and biases
    def tune(self, wfactors, bfactors):
        factor = self.LR
        for l in range(0, len(self.layers) - 1):
            self.weights[l] = np.subtract(self.weights[l], np.multiply(factor, wfactors[l]))
            self.biases[l] = np.subtract(self.biases[l], np.multiply(factor, bfactors[l]))
    '''


    def readLabels(self, filePath):
        with gzip.open(filePath, 'rb') as f:
            magic_numb = int.from_bytes(f.read(4), 'big')
            label_numb = int.from_bytes(f.read(4), 'big')

            labels = np.frombuffer(f.read(self.TrainNum), dtype=np.uint8)


            vectorized_labels = np.zeros((labels.size, 10))

            #Make the number at the correspoinding position in the vector 1
            for i in range(0,labels.size):
                vectorized_labels[i][labels[i]] = 1

        return vectorized_labels


#Get training images
    def readTraining(self, filePath):
        with gzip.open(filePath, 'rb') as f:

            magic_numb = int.from_bytes(f.read(4), 'big')
            image_numb = int.from_bytes(f.read(4), 'big')
            row_numb = int.from_bytes(f.read(4), 'big')
            col_numb = int.from_bytes(f.read(4), 'big')

            ##Get image data
            raw_images = np.frombuffer(f.read(), dtype=np.uint8)
            shaped_images = raw_images.reshape((image_numb, row_numb, col_numb))

            ##Normalize and Flatten the data

            # Normalize the data
            shaped_images = shaped_images / 255.0

            shaped_images = shaped_images[:self.TrainNum]

            # Flatten the data
            shaped_images = shaped_images.reshape(self.TrainNum, -1)

        return shaped_images

#Format labels and images into pairs
    def getFormattedTraining(self, imagesPath, labelPath):

        images = self.readTraining(imagesPath)
        labels = self.readLabels(labelPath)

        training_set = [[image, label] for image, label in zip(images, labels)]
    
        return training_set




    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.where(x <= 0, 0, 1)