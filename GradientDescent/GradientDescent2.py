#
# Gradient descent for multivariate linear regression
#
import numpy as np
import matplotlib.pyplot as plt

# Features normalization
def featureNormalize(x):
    avg = np.mean(x, axis=0)
    sig = np.std(x, axis=0)
    x_norm = np.divide(np.subtract(x, avg),sig)
    return x_norm

# Load data file
filenameIn = 'TwoFeatures.csv'
data = np.loadtxt(filenameIn, delimiter=',')
print('Shape of data', data.shape)

# Extract x (features) matrix and y (target) vector from data
print('---Extract data---')
x = data[:,:-1]
y = data[:,-1]
print('Shape of x', x.shape)
print('Shape of y', y.shape)

# Normalize features
print('---Normalize features---')
x_norm = featureNormalize(x)
print('Average of x_norm', np.mean(x_norm))
print('Stddev of x_norm', np.std(x_norm))
print('Shape of x_norm', x_norm.shape)

# Reshape y
print('---Reshape matrices---')
y = y.reshape(-1,1) # vector
print('Shape of y', y.shape)

# Construct X matrix
onesVect = np.ones((x_norm.shape[0],1), dtype=int)
X = np.concatenate((onesVect, x_norm), axis=1)
print('Shape of X', X.shape)

# Initialize theta vector
print('---Initialize theta and perform gradient descent---')
theta = np.zeros((X.shape[1],1))
print('Shape of theta vector', theta.shape)

# define descent rate and number of iterations
alpha = 0.03
nIter = 700
m = y.size

# Gradient descent loop
jTheta = []
for i in range(1, nIter):
    t1 = X.dot(theta)
    t2 = np.subtract(t1, y)
    XT = X.T
    t3 = XT.dot(t2)

    # Update theta
    temp = theta - alpha / m * t3
    theta = temp

    # Cost function
    jThetaLoop = 0.5 / m * t2.T.dot(t2)
    jTheta.append(jThetaLoop[0,0])

# Solution of gradient descent
for i in range(theta.shape[0]):
    print('theta_'+str(i) + ' = ' + str(theta[i,0]))

plt.plot(jTheta, marker='.')
plt.xlabel('Iteration number')
plt.ylabel(r'Cost function J($\theta$)')
plt.show()

# Solution from normal equation
print('---Solution from normal equation---')
thetaNormal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
for i in range(thetaNormal.shape[0]):
    print('theta_'+str(i) + ' = ' + str(thetaNormal[i,0]))
