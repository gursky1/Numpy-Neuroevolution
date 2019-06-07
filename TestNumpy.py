import numpy as np 


# Defining the softmax function
def softmax(x):
    x = np.exp(x - np.max(x))
    return np.array(x / x.sum())


# Defining our activation function
def relu(x):
    return np.where(x>0,x,0)

#inputs = np.array([[1,5,2],[1,3,5]])
inputs = np.random.uniform(-1,1,size=(1000,3))
W = np.random.uniform(-1,1,size=(3,3))
bias = np.repeat(1,3)
a = relu((inputs@W)+bias)
a = np.apply_along_axis(softmax, axis=1, arr=a)
print(a)
result = np.argmax(a,axis=1)
print(result)