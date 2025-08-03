import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from lr_utils import load_dataset
# load dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# if you want to check the pictures , use code below & change the costs plot to comment in the last line of code:

# index =3
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[0, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# plt.show()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])

train_set_x_flatten = train_set_x_flatten/255.
test_set_x_flatten = test_set_x_flatten/255.

# implementation of logistic regression 
# initializing parameters w,b
w = np.zeros((train_set_x_flatten.shape[0] , 1))
b = 0
m_train = train_set_x_orig.shape[0]
m_test = test_set_y.shape[1]
costs=[]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def prop(w,b):
    for i in range(500):
        z= np.dot(w.T , train_set_x_flatten) + b 
        a = sigmoid(z)
        dz = a - train_set_y
        dw = (1/m_train) * np.dot(train_set_x_flatten , dz.T)
        db = (1/m_train) * np.sum(dz)
        w -= (0.005)*dw
        b -= (0.005)*db 
        cost = -1/m_train * np.sum(train_set_y*np.log(a) + (1-train_set_y)*np.log(1-a))
        if i % 10 == 0:
            costs.append(cost)

    return(w,b)       

w , b = prop(w,b)

def predict(w, b, X):
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    y_pred = a > 0.5
    return y_pred.astype(int)
train_pred = predict(w,b,train_set_x_flatten)
test_pred = predict(w,b,test_set_x_flatten)


print()
print("train accuracy: {} %".format(100 - np.mean(np.abs(train_pred - train_set_y)) * 100))
print()
print("test accuracy: {} %".format(100 - np.mean(np.abs(test_pred - test_set_y)) * 100))


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()

