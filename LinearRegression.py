import random
import numpy as np
import matplotlib.pyplot as plt

#Get array of vectors
#choose random m and b values
#compute derivative for point at m
#compute derivative for point at n
#m = m - a*dF/dm
#same for b
#Assume convergence at 10 iterations or when If dF/dm = 0
#return final m and b values

rand_m = 0.3
rand_b = 0.1

##Returns dF/dm
def with_m(array_of_vectors,m,b):

    sum = 0

    for i in range(len(array_of_vectors)):
        sum += (array_of_vectors[i][1] - (m*array_of_vectors[i][0] + b)) * array_of_vectors[i][0]

    return sum/len(array_of_vectors)

def with_b(array_of_vectors,m,b):

    sum = 0

    for i in range(len(array_of_vectors)):
        sum += array_of_vectors[i][1] - (m*array_of_vectors[i][0] + b)

    return sum/len(array_of_vectors)

def generate_line(array_of_vectors):
    iterations = 1000
    alpha = 0.000001
    m = 1.5 ##They get returned as Nan??
    b = 1.5

    global rand_m
    rand_m = m
    global rand_b
    rand_b = b

    for i in range(iterations):
        dM = with_m(array_of_vectors,m,b)
        dB = with_b(array_of_vectors,m,b)

        m -= alpha*dM
        b -= alpha*dB

    return [m,b]

x = np.array([0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16])

data_plot = [[0,0],[1,2],[5,6],[10,20],[8,8]]

coefs = generate_line(data_plot)

y_line = coefs[0]*x + coefs[1]

z, q = zip(*data_plot)

# Plotting the points
plt.scatter(z, q, color='blue', label='Data Points')
plt.plot(x, y_line, color='red', label=f'y = {coefs[0]}x + {coefs[1]}')  # Line plot for y = mx + b

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Data Points with Line')
plt.legend()
plt.grid(True)
plt.show()

print(str(rand_m)+" / "+str(rand_b))
print(str(coefs[0])+" / "+str(coefs[1]))

    