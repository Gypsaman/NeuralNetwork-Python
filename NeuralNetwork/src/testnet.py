import numpy as np
import math
from sklearn.utils import shuffle

#class LROptimizerType(object):
# NONE = 1
# ADAM = 2
#class BatchNormMode(object):
# TRAIN = 1
# TEST = 2
#class ActivationType(object):
# SIGMOID = 1
# TANH = 2
# RELU = 3
# SOFTMAX = 4

class Layer(object):
 def __init__(self,numNeurons,numNeuronsPrevLayer, batchsize,lastLayer= False,dropOut= 0.2,activationType=ActivationType.SIGMOID):
 # initialize the weights and biases
     self.numNeurons = numNeurons
     self.batchsize = batchsize
     self.lastLayer = lastLayer
     self.numNeuronsPrevLayer = numNeuronsPrevLayer
     self.activationFunction = activationType
     self.dropOut = dropOut

 #self.W = 0.01 * np.random.randn(numNeurons,numNeuronsPrevLayer)
     self.W = np.random.uniform(low=-0.1,high=0.1,size=(numNeurons,numNeuronsPrevLayer))
     self.b = np.random.uniform(low=-1,high=1,size=(numNeurons))
     self.WGrad = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.bGrad = np.zeros((numNeurons)) # gradient for delta
 #-------------following for batch norm--------------
     self. mu = np.zeros((numNeurons)) # batch mean
     self.sigma2 = np.zeros((numNeurons)) # sigma^2 for batch
     self.epsilon = 1e-6
     self.gamma = np.random.rand(1)
     self.beta= np.random.rand(1)
     self.S = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.Shat = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.Sb = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.runningmu = np.zeros((numNeurons))
     self.runningsigma2 = np.zeros((numNeurons))
     self.dgamma = np.zeros((numNeurons))
     self.dbeta = np.zeros((numNeurons))
     self.delta = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.deltabn = np.zeros((numNeurons,numNeuronsPrevLayer))
 #---------------------------------------------------

 #----------following for implementing ADAM-----------
     self.mtw = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.mtb = np.zeros((numNeurons))
     self.vtw = np.zeros((numNeurons,numNeuronsPrevLayer))
     self.vtb = np.zeros((numNeurons))
 #----------------------------------------------------
     self.zeroout = None # for dropout

 def Evaluate(self,indata, doBatchNorm=False, batchMode= BatchNormMode.TRAIN):
     self.S = np.dot(indata,self.W.T) + self.b
     if (doBatchNorm == True):
         if (batchMode == BatchNormMode.TRAIN):
             self.mu = np.mean(self.S, axis=0) # batch mean
             self.sigma2 = np.var(self.S,axis=0) # batch sigma^2
             self.runningmu = 0.9 * self.runningmu + (1 - 0.9)* self.mu
             self.runningsigma2 = 0.9 * self.runningsigma2 + (1 - 0.9)* self.sigma2
         else:
             self.mu = self.runningmu
             self.sigma2 = self.runningsigma2
         self.Shat = (self.S - self.mu)/np.sqrt(self.sigma2 + self.epsilon)
         self.Sb = self.Shat * self.gamma + self.beta
         sum = self.Sb
     else:
        sum = self.S
     if (self.activationFunction == ActivationType.SIGMOID):
         self.a = self.sigmoid(sum)
         self.derivAF = self.a * (1 - self.a)
     if (self.activationFunction == ActivationType.TANH):
         self.a = self.TanH(sum)
         self.derivAF = (1 - self.a*self.a)
     if (self.activationFunction == ActivationType.RELU):
         self.a = self.Relu(sum)
     #self.derivAF = 1.0 * (self.a > 0)
     epsilon=1.0e-6
     self.derivAF = 1. * (self.a > epsilon)
     self.derivAF[self.derivAF == 0] = epsilon
     if (self.activationFunction == ActivationType.SOFTMAX):
         self.a = self.Softmax(sum)
         self.derivAF = None # we do delta computation for Softmax layer in Network
     if (self.lastLayer == False):
         self.zeroout = np.random.binomial(1,self.dropOut,(self.numNeurons))/self.dropOut
         self.a = self.a * self.zeroout
         self.derivAF = self.derivAF * self.zeroout
 
 def sigmoid(self,x):
    return 1 / (1 + np.exp(-x)) # np.exp makes it operate on entire array

 def TanH(self, x):
    return np.tanh(x)
 def Relu(self, x):
    return np.maximum(0,x)
 def Softmax(self, x):
     if (x.shape[0] == x.size):
        ex = np.exp(x)
        return ex/ex.sum()
     ex = np.exp(x)
     for i in range(ex.shape[0]):
         denom = ex[i,:].sum()
         ex[i,:] = ex[i,:]/denom
     return ex

class Network(object):
    def __init__(self,X,Y,numLayers,batchsize,dropOut = 1.0,activationF=ActivationType.SIGMOID, lastLayerAF= ActivationType.SIGMOID):
         self.X = X
         self.Y = Y
         self.batchsize = batchsize
         self.numLayers = numLayers
         self.Layers = [] # network contains list of layers
         self.lastLayerAF = lastLayerAF
         for i in range(len(numLayers)):
             if (i == 0): # first layer
                layer = Layer(numLayers[i],X.shape[1],batchsize,False,dropOut,activationF)
             elif (i == len(numLayers)-1): # last layer
                layer = Layer(Y.shape[1],numLayers[i-1],batchsize,True,dropOut,lastLayerAF)
             else: # intermediate layers
                layer = Layer(numLayers[i],numLayers[i-1],batchsize,False,dropOut,activationF)
             self.Layers.append(layer);

    def Evaluate(self,indata,doBatchNorm=False,batchMode=BatchNormMode.TEST): #evaluates all layers
         self.Layers[0].Evaluate(indata, doBatchNorm,batchMode) # first layer
         for i in range(1,len(self.numLayers)):
            self.Layers[i].Evaluate(self.Layers[i-1].a,doBatchNorm,batchMode)
         return self.Layers[len(self.numLayers)-1].a
 
    def Train(self, epochs,learningRate, lambda1, batchsize=1,LROptimization=LROptimizerType.NONE,doBatchNorm=False):
         itnum = 0
         for j in range(epochs):
            error = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            for i in range(0, self.X.shape[0], batchsize):
            # get (X, y) for current minibatch/chunk
                 X_train_mini = self.X[i:i + batchsize]
                 y_train_mini = self.Y[i:i + batchsize]
                 self.Evaluate(X_train_mini,doBatchNorm,batchMode=BatchNormMode.TRAIN)
                 if (self.lastLayerAF == ActivationType.SOFTMAX):
                    error += -(y_train_mini * np.log(self.Layers[len(self.numLayers)-1].a+0.001)).sum()
                 else:
                    error += ((self.Layers[len(self.numLayers)-1].a - y_train_mini) * (self.Layers[len(self.numLayers)-1].a - y_train_mini)).sum()
            
            lnum = len(self.numLayers)-1 # last layer number
             # compute deltas, grads on all layers
            while(lnum >= 0):
                 if (lnum == len(self.numLayers)-1): # last layer
                     if (self.lastLayerAF == ActivationType.SOFTMAX):
                         self.Layers[lnum].delta = -y_train_mini + self.Layers[lnum].a
                     else:
                         self.Layers[lnum].delta = -(y_train_mini-self.Layers[lnum].a) * self.Layers[lnum].derivAF
                 else: # intermediate layer
                    self.Layers[lnum].delta = np.dot(self.Layers[lnum+1].delta,self.Layers[lnum+1].W) * self.Layers[lnum].derivAF

                 if (lnum > 0): #previous output
                    prevOut = self.Layers[lnum-1].a
                 else:
                    prevOut = X_train_mini

                 if (doBatchNorm == True):
                    self.Layers[lnum].dbeta = np.sum(self.Layers[lnum].delta,axis=0)
                    self.Layers[lnum].dgamma = np.sum(self.Layers[lnum].delta * self.Layers[lnum].Shat,axis=0)
                    self.Layers[lnum].deltabn = (self.Layers[lnum].delta * self.Layers[lnum].gamma)/(batchsize*np.sqrt(self.Layers[lnum].sigma2 +self.Layers[lnum].epsilon )) * (batchsize -1 - (self.Layers[lnum].Shat * self.Layers[lnum].Shat))

                    self.Layers[lnum].WGrad = np.dot(self.Layers[lnum].deltabn.T,prevOut)
                    self.Layers[lnum].bGrad = self.Layers[lnum].deltabn.sum(axis=0)
                 else:
                    self.Layers[lnum].WGrad = np.dot(self.Layers[lnum].delta.T,prevOut)
                    self.Layers[lnum].bGrad = self.Layers[lnum].delta.sum(axis=0)
                 lnum = lnum - 1
                 itnum = itnum + 1
                 self.UpdateGradsBiases(learningRate,lambda1, batchsize, LROptimization,itnum, doBatchNorm)
            print("Iter = " + str(j) + " Error = "+ str(error))

    def UpdateGradsBiases(self, learningRate, lambda1, batchSize, LROptimization, itnum,doBatchNorm):
     # update weights and biases for all layers
         beta1 = 0.9
         beta2 = 0.999
         epsilon = 1e-8
         for ln in range(len(self.numLayers)):
             if (LROptimization == LROptimizerType.NONE):
                 self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * self.Layers[ln].WGrad - learningRate * lambda1 * self.Layers[ln].W.sum()
                 self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * self.Layers[ln].bGrad
             elif (LROptimization == LROptimizerType.ADAM):
                 gtw = self.Layers[ln].WGrad
                 gtb = self.Layers[ln].bGrad
                 self.Layers[ln].mtw = beta1 * self.Layers[ln].mtw + (1 - beta1) * gtw
                 self.Layers[ln].mtb = beta1 * self.Layers[ln].mtb + (1 - beta1) * gtb
                 self.Layers[ln].vtw = beta2 * self.Layers[ln].vtw + (1 - beta2) * gtw*gtw
                 self.Layers[ln].vtb = beta2 * self.Layers[ln].vtb + (1 - beta2) * gtb*gtb
                 mtwhat = self.Layers[ln].mtw/(1 - beta1**itnum)
                 mtbhat = self.Layers[ln].mtb/(1 - beta1**itnum)
                 vtwhat = self.Layers[ln].vtw/(1 - beta2**itnum)
                 vtbhat = self.Layers[ln].vtb/(1 - beta2**itnum)
                 self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * mtwhat /((vtwhat**0.5) + epsilon)
                 self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * mtbhat /((vtbhat**0.5) + epsilon)
             if (doBatchNorm == True):
                self.Layers[ln].beta = self.Layers[ln].beta - learningRate * self.Layers[ln].dbeta
                self.Layers[ln].gamma = self.Layers[ln].gamma - learningRate * self.Layers[ln].dgamma
