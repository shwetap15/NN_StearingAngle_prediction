## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
from tqdm import tqdm
import time
import math

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    csv_size = data.shape[0]
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    # You could import your images one at a time or all at once first, 
    # here's some code to import a single image:
    
    image_size = (64,60)
    num_train_images = csv_size

    num_channels = 1
    train_x = np.zeros((num_train_images,(image_size[0]*32*num_channels)))
    train_y = np.zeros((num_train_images,1))
    
    count1 = 0

    train_labels = []
    
    for i in range(num_train_images):
        frame_num = int(frame_nums[i])
        im_full = cv2.imread(path_to_images+ '/' + str(int(frame_nums[i])).zfill(4) + '.jpg')
        im_full = cv2.resize(im_full,(64,60))
        im_full = im_full[28:]
        processed_im = processImg(im_full)
        
        processed_im = processed_im/255
        
        x = processed_im.flatten()
        x = np.expand_dims(x, axis=0)
        
        train_x[count1,:] = x
        train_y[count1,:] = round(steering_angles[i],3)
        train_labels.append(round(steering_angles[i],3))
        count1+=1
    
    # Train your network here. You'll probably need some weights and gradients!
    NN = NeuralNetwork()
    #params = NN.getParams()
    #grads = NN.computeGradients(X, y)

    num_iterations = 7000
    alpha = 1e-4
    beta1= 0.9
    beta2= 0.999
    epsilon= 1e-08


    NN.min_angle = np.min(train_y) 
    NN.max_angle = np.max(train_y) 


    NN.min_max_angle = abs(NN.max_angle - NN.min_angle)
    train_y = (train_y+abs(NN.min_angle))/NN.min_max_angle

    grads = NN.computeGradients(train_x, train_y)

    m0 = np.zeros(len(grads)) #Initialize first moment vector
    v0 = np.zeros(len(grads)) #Initialize second moment vector
    t = 0.0

    losses = [] #For visualization
    mt = m0
    vt = v0

    for i in tqdm(range(num_iterations)):
        t += 1
        grads = NN.computeGradients(train_x, train_y)
        mt = beta1*mt + (1-beta1)*grads
        vt = beta2*vt + (1-beta2)*grads**2
        mt_hat = mt/(1-beta1**t)
        vt_hat = vt/(1-beta2**t)
        
        params = NN.getParams()
        new_params = params - alpha*mt_hat/(np.sqrt(vt_hat)+epsilon)
        NN.setParams(new_params)
        
        losses.append(NN.costFunction(train_x, train_y))
    
    #plot(losses)
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file)

    ## Perform inference using your Neural Network (NN) here.
    im_full = cv2.resize(im_full,(64,60))
    im_full = im_full[28:]
    processed_im = processImg(im_full)
    
    processed_im = processed_im/255
    
    x = processed_im.flatten()
    x = np.expand_dims(x, axis=0)

    predicted = NN.predict(x)
    predicted_val = float(predicted)
    
    predicted_val = predicted_val*NN.min_max_angle
    predicted_val = predicted_val - abs(NN.min_angle)
    predicted_val = predicted_val
    
    return predicted_val

class NeuralNetwork(object):
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 32*64
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5
        
        self.min_angle = 0
        self.max_angle = 0
        self.min_max_angle = 0
        #Weights (parameters)
        # Glorot Initialization
        limit = np.sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))

        limit = np.sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
    def predict(self,X):
        return float(self.forward(X))
        
def processImg(image):
    original_img = image
    width = image.shape[1]
    height = image.shape[0]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image