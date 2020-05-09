#
# Gradient descent with 1D data i.e. 2 features
#
import numpy as np
import matplotlib.pyplot as plt

# Load data file
filenameIn = 'OneFeature.csv'
data = np.loadtxt(filenameIn, delimiter=',')
print('Shape of data matrix', data.shape)

# Extract x and y vectors from data
x = data[:,0]
y = data[:,1]

# Reshape to have them in column vector style
x = x.reshape(-1,1)
y = y.reshape(-1,1)
print('Shape of target vector', y.shape)

# Construct X matrix
onesVect = np.ones((x.shape[0],1), dtype=int)
X = np.concatenate((onesVect, x), axis=1)
print('Shape of X matrix', X.shape)

# Initialize theta vector
theta = np.zeros((2,1))
print('Shape of theta vector', theta.shape)

# define descent rate and number of iterations
alpha = 0.01
nIter = 1500
m = y.size

#Gradient descent loop
jTheta = []
for i in range(1, nIter):
    t1 = X.dot(theta)
    t2 = np.subtract(t1, y)
    XT = np.transpose(X)
    t3 = XT.dot(t2)

    temp = theta - alpha / m * t3
    theta = temp

    jThetaLoop = 0.5 / m * np.transpose(t2).dot(t2)
    jTheta.append(jThetaLoop[0,0])

print('theta_0 = ' + str(theta[0,0]))
print('theta_1 = ' + str(theta[1,0]))

plt.plot(jTheta, marker='.')
plt.show()