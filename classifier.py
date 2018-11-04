import numpy as np
def sigmoid(x, deriv = False):
    if deriv == False:
        return 1 / (1 + np.e**(-x))
    else:
        return sigmoid(x) * (1 - sigmoid(x))

def prob(x):
    for a in range(x.shape[0]):
        if x[a] >= 0.5:
            x[a] = 1
        else:
            x[a] = 0
    return x

class NeuralNetwork():
    
    def __init__(self, n_features, n_hidden):
        '''n_features: Number of features in a data sample.
           n_hidden: Number of hidden layer units.'''
        self.w1 = np.random.randn(n_features, n_hidden)
        self.b1 = np.random.randn(n_hidden)
        self.w2 = np.random.randn(n_hidden, 1)
        self.b2 = np.random.randn(1)
        self.error = None
        
    def forward(self, X):
        self.X = X
        self.z = X.dot(self.w1) + self.b1
        self.ah = sigmoid(self.z)
        self.out = self.ah.dot(self.w2) + self.b2
        self.aout = sigmoid(self.out)
        
    def backprop(self, Y):
        self.error = -(Y * np.log(self.aout) + ((1 - Y) * np.log(1 - self.aout)))
        delout = ((Y/self.aout) - ((1-Y)/(1-self.aout))) * sigmoid(self.out, True)
        errorh = delout.dot(self.w2.T)
        delh = errorh * sigmoid(self.z, True)
        self.w2 += 0.001 * self.ah.T.dot(delout)
        self.w1 += 0.001 * self.X.T.dot(delh)
        self.b2 += 0.001 * np.sum(delout, 0)
        self.b1 += 0.001 * np.sum(delh, 0)
        
    def predict(self):
        self.predictions = prob(self.aout)
        return self.predictions

D = 2 # dimensionality of input
    
X1 = np.random.randn(2, D) + np.array([0, -2])
X2 = np.random.randn(2, D) + np.array([2, 2])
X3 = np.random.randn(2, D) + np.array([0, -2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*2 + [1]*2 + [0]*2)
model = NeuralNetwork(2,6)
Y = Y.reshape((6,1))
for t in range(10000):
    if(t % 1000 == 0):
        print("Cost: {} after {} epochs.".format(np.sum(model.error), t))
    model.forward(X)
    model.backprop(Y)
proba = prob(model.aout)
print("Predictions of the model are:")
print(proba)
