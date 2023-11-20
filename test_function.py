import numpy as np
from BabyNetwork import BabyNetwork
from MatureNetwork import MatureNetwork
import matplotlib.pyplot as plt
import numpy as np





helper = BabyNetwork(300,[764,30,10])
solver = MatureNetwork() #98% For the config!!!


#Returns both the flattened image, and the expected vector
test_image = helper.getFormattedTraining('./MNIST dataset/t10k-images-idx3-ubyte.gz', './MNIST dataset/t10k-labels-idx1-ubyte.gz')[7]


guessed = solver.guessNumber(test_image[0])

correct_digit = next((i for i, x in enumerate(test_image[1]) if x), None)

if(guessed == correct_digit):
    print('SOLVED! Digit: '+str(guessed))
else:
    print('NOPE Guessed: '+str(guessed)+' was '+str(correct_digit))



'''
with open('NetworkConfig.txt', 'w') as txt_file:
    txt_file.write(' '.join(np.char.mod('%d', config))+'\n\n\n\n')

    for layer in weights:
        for neuron in layer:
            txt_file.write(' '.join(np.char.mod('%d', neuron))+'\n')
        txt_file.write('\n\n')

    txt_file.write('\n')
    
    for layer in biases:
        txt_file.write(' '.join(np.char.mod('%d', layer))+'\n')
'''

