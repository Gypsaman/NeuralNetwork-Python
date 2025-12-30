import math
from ActivationType import ActivationType
import numpy as np
from scipy import signal as sg
from timeit import default_timer as timer

class LROptimizerType(object):
    NONE = 1
    ADAM = 2
def loss(predict,y,AF):
    #if AF == ActivationType.SOFTMAX:
    #    Loss = -(y*np.log(predict+0.001)).sum()
    #else:
    #    Loss = (0.5 * (np.multiply((predict-y),(predict-y)))).sum()/predict.shape[1]
    Loss = np.multiply((predict-y),(predict-y)).sum()/predict.shape[0]
    return Loss

def relu(data):
    sh = data.shape
    result = np.array([[x if x > 0 else 0 for x in row] for row in data]).reshape(sh)
    return result

def bprelu(dA,activation):
    indim = 1
    for dim in dA.shape:
        indim *= dim
    tmp = dA.reshape(indim)
    indices = activation.reshape(indim)
    tmp[indices<0] = 0
    result = np.array(tmp).reshape(dA.shape)
    return result

def sigmoid(data):
    result = 1/(1+np.exp(-1*data))
    return result

def bpsigmoid(dA,activation):
    result = dA*(activation*(1-activation))
    return result

def tanh(data):
    result = (np.exp(data)-np.exp(-1*data)/(np.exp(data)+np.exp(-1*data)))
    return result

def bptanh(dA,activation):
    result = dA*(1-activation*activation)
    return result

def softmax(data):
    if data.shape[0] == data.size:
        esum = np.exp(data).sum()
        result = np.exp(data)/esum
        return result
    ex = np.exp(data)
    for i in range(data.shape[0]):
        denom = ex[i,:].sum()
        ex[i,:] = ex[i,:]/denom
    return ex

def bpsoftmax(dA,activation): 
    result = dA
    return result

Activations = [sigmoid,tanh,relu,softmax]
Gradients = [bpsigmoid,bptanh,bprelu,bpsoftmax]

def convolvexx(mat,kern):
    (mH,mW) = mat.shape
    ks = kern.shape[0]
    kern = np.rot90(kern,k=2)
    output = np.zeros((mH-ks+1,mW-ks+1))
    for row in np.arange(0,mH-ks+1):
        for col in np.arange(0,mW-ks+1):
            subm = mat[row:row+ks,col:col+ks]
            k = (subm * kern).sum()
            output[row,col] = k
    return output
def fullconvolvexx(mat,kern):
    (rows,cols) = mat.shape
    ks = kern.shape[0]
    #kern = np.rot90(kern,k=2)
    output = np.zeros((rows+ks-1,cols+ks-1))
    for row in range(rows+ks-1):
        for col in range(cols+ks-1):
            sum = 0
            for ki in np.arange(-(ks-1),1):
                for kj in np.arange(-(ks-1),1):
                    if (row+ki) >= 0 and (row +ki) < rows and (col + kj) >= 0 and (col+kj) < cols:
                        data = mat[row+ki,col+kj]
                        kval = kern[ki+(ks-1),kj+(ks-1)]
                        sum += data * kval;
            output[row,col] = sum
    return output


class DenseLayer(object):
    """description of class"""
    numparams = 2
    epsilon = 1.0e-9
    AdamBeta1 = 0.9
    AdamBeta2 = 0.999
    
    def __init__(self, neurons, inputs, activationf=ActivationType.RELU, BatchNorm = False):

        self.layerNodes = neurons
        self.inputNodes = inputs

        self.weights = np.random.uniform(low=-0.1,high=0.1,size=(neurons,inputs))
        self.biases = np.zeros((neurons))
        self.activationf = activationf
        self.prev_a = np.zeros((inputs))
        self.delta = np.zeros((neurons))
        self.dw = np.zeros(neurons)
        self.db = np.zeros(neurons)
        self.activation = np.zeros((neurons))

        # batchNorm
        self.isBatchNorm = BatchNorm
        self.BNBeta = np.random.rand(1)
        self.BNgamma = np.random.rand(1)
        self.Shat = np.zeros((neurons,inputs))
        self.Sb = np.zeros((neurons,inputs))
        self.mu = np.zeros((neurons))
        self.sigma2 = np.zeros((neurons))
        self.gamma = np.random.rand(1)
        self.beta = np.random.rand(1)
        self.runningmu = np.zeros((neurons))
        self.runningsigma2 = np.zeros((neurons))

        #Adam Variables
        self.mt = np.zeros((DenseLayer.numparams,neurons))
        self.vt = np.zeros((DenseLayer.numparams,neurons))
        
    def forward(self,prev_activation,isTraining = True):
        assert prev_activation.shape[1] == self.inputNodes

        self.prev_a = prev_activation.astype(float)
        s = np.dot(self.prev_a,np.transpose(self.weights)) + self.biases
        if self.isBatchNorm == True:
            if(isTraining):
                self.mu = np.mean(s,axis=1).reshape((self.layerNodes,1))
                self.sigma2 = np.var(s,axis=1).reshape((self.layerNodes,1))
                self.runningmu = 0.9 * self.runningmu + (1-0.9) * self.mu
                self.runningsigma2 = 0.9 * self.runningsigma2 + (1-0.9) * self.sigma2
            else:
                self.mu = self.runningmu
                self.sigma2 = self.runningsigma2

            self.Shat = (s - self.mu)/np.sqrt(self.sigma2 + DenseLayer.epsilon)
            self.Sb = self.Shat * self.gamma + self.beta
            s = self.Sb

        self.activation = Activations[self.activationf](s)

        return self.activation

    def backprop(self,dA):
        m = self.prev_a.shape[0]
        
        self.delta = Gradients[self.activationf](dA,self.activation)
        
        if self.isBatchNorm == True:
            self.dbeta = np.sum(self.delta,axis=1).reshape((self.layerNodes,1))
            self.dgamma = np.sum(self.delta*self.Shat,axis=1).reshape((self.layerNodes,1))
            self.deltabn = (self.delta * self.gamma)/(m*np.sqrt(self.sigma2+DenseLayer.epsilon)) * (m-1-(self.Shat*self.Shat))

            self.dw = 1/m*np.dot(self.deltabn,self.prev_a.T)
            self.db = 1/m*np.sum(self.deltabn,axis=1,keepdims=True)

        else:
            self.dw = 1/m*np.dot(self.delta.T,self.prev_a)
            self.db = 1/m*np.sum(self.delta,axis=0)
        da_prev = np.dot(self.delta,self.weights)

        return da_prev

    def Adam(self,t):

        #Adam calc for W
        self.mt[0] = DenseLayer.AdamBeta1*self.mt[0] + (1-DenseLayer.AdamBeta1)*self.dw
        self.vt[0] = DenseLayer.AdamBeta2*self.vt[0] + (1-DenseLayer.AdamBet2)*self.dw*self.dw
        mtBiased = self.mt[0]/(1-np.power(DenseLayer.AdamBeta1,t))
        vtBiased = self.vt[0]/(1-np.power(DenseLayer.AdamBeta2,t))
        self.dw = mtBiased *(1/(np.sqrt(vtBiased)-DenseLayer.epsilon))
        #Adam calc for b
        self.mt[1] = DenseLayer.AdamBeta1*self.mt[1] + (1-DenseLayer.AdamBeta1)*self.db
        self.vt[1] = DenseLayer.AdamBeta2*self.vt[1] + (1-DenseLayer.AdamBeta2)*self.db*self.db
        mtBiased = self.mt[1]/(1-np.power(DenseLayer.Layer.AdamBeta1,t))
        vtBiased = self.vt[1]/(1-np.power(DenseLayer.AdamBeta2,t))
        self.db = mtBiased *(1/(np.sqrt(vtBiased)-DenseLayer.epsilon))                               

        da_prev = np.dot(self.weights.T,self.delta)
        return da_prev

    def dump(self):
        np.savetxt('c:\import\W'+str(self.layerNodes)+'-'+str(self.inputNodes)+'.csv',self.weights,delimiter=',')
        np.savetxt('c:\import\B'+str(self.layerNodes)+'-'+str(self.inputNodes)+'.csv',self.biases,delimiter=',')

    def update_parameters(self,lr,optimizer = LROptimizerType.NONE,iternum=0):
        if optimizer == LROptimizerType.NONE:
            self.weights -= lr * self.dw
            self.biases -=  lr * self.db
        elif optimizer == LROptimizerType.ADAM:
            self.Adam(iternum)
        if self.isBatchNorm == True:
            self.beta = self.beta - lr * self.dbeta
            self.gamma = self.gamma - lr * self.dgamma

    def GetSize(self):
        return self.layerNodes
    def GetType(self):
        return "Dense"

class CNNLayer(object):

    def __init__(self,PrevInput,features,filtersize,activation=ActivationType.RELU):
        self.InputH,self.InputW,self.InputC = PrevInput
        self.features = features
        self.inputfeatures = self.InputC
        self.featureH = self.InputH - filtersize + 1
        self.featureW = self.InputW - filtersize + 1

        uniform = math.sqrt(features/((self.inputfeatures + self.features)*filtersize*filtersize))
        self.kernelsize = (self.features,self.inputfeatures,filtersize,filtersize)
        self.kernels = np.random.uniform(low=-uniform,high=uniform,size=self.kernelsize) * 0.1

        self.biases = np.zeros(features)
        self.activationf = activation

        self.gradk = np.zeros(self.kernelsize)
        self.gradb = np.zeros(self.features)

    def initkernels(self):
        for feature in range(self.features):
            for input in range(self.inputfeatures):
                self.kernels[feature,input] = np.loadtxt('c:/import/k'+str(self.inputfeatures)+'-'+str(self.features)+'-'+str(input)+'-'+str(feature)+'.csv',delimiter=',')
        return
    def forward(self,prev_activation,isTraining = True):
        self.prev_a = prev_activation
        self.samples = prev_activation.shape[0]
        self.Cc = np.zeros((self.samples,self.featureH,self.featureW,self.features))
        self.C1 = np.zeros((self.samples,self.featureH,self.featureW,self.features))


        for sample in range(self.samples):
            for feature in range(self.features):
                for input in range(self.inputfeatures):
                    self.Cc[sample,:,:,feature] += sg.convolve2d(self.prev_a[sample,:,:,input],self.kernels[feature,input,:,:],'valid') 
            
        for feature in range(self.features):
            self.Cc[sample,:,:,feature] += self.biases[feature]
            self.C1[sample,:,:,feature] = Activations[self.activationf](self.Cc[sample,:,:,feature].reshape(1,self.featureH*self.featureW)).reshape((self.featureH,self.featureW))
            
        return self.C1

    def backprop(self,dA):
        self.gradk = np.zeros(self.kernelsize)
        self.gradb = np.zeros(self.features)
        self.delta = np.zeros(dA.shape)
        output = np.zeros(self.prev_a.shape)
        self.delta = Gradients[self.activationf](dA,self.Cc)

        for sample in range(self.samples):
            for input in range(self.inputfeatures):
                
                for feature in range(self.features):
                    val = sg.convolve2d(np.rot90(self.prev_a[sample,:,:,input],2),self.delta[sample,:,:,feature],'valid')
                    self.gradk[feature,input] += val

                self.gradb[feature] += dA[sample,:,:,feature].sum()
                output[sample,:,:,input] = sg.convolve2d(self.delta[sample,:,:,feature],np.rot90(self.kernels[feature,input],2),'full')
                

        return output

    def dump(self):
        for feature in range(self.features):
            for input in range(self.inputfeatures):
                np.savetxt('c:/import/gradk-f'+str(self.features)+'-I'+str(self.inputfeatures)+'-'+str(feature)+'-'+str(input)+'.csv',self.gradk[feature,input],delimiter=',')
        np.savetxt('c:/import/bias-f'+str(self.features)+'-I'+str(self.inputfeatures)+'.csv',self.biases,delimiter=',')

    def update_parameters(self,lr,optimizer = LROptimizerType.NONE,iternum=0):
        for feature in range(self.features):
            for input in range(self.inputfeatures):
                self.kernels[feature,input] -= lr * self.gradk[feature,input]/self.samples
            self.biases[feature] -= lr * self.gradb[feature]/self.samples
        return

    def GetSize(self):
        size = (self.featureH,self.featureW,self.features)
        return size

    def GetType(self):
        return "Convolutional "

class FlattenLayer(object):

    def __init__(self,PrevInput):
        self.PrevInput = PrevInput
        self.OutDim = 1
        
        for val in self.PrevInput:
            self.OutDim *= val
        #self.activation = np.zeros(prev_activation.shape[0],self.OutDim)

    def forward(self,prev_activation,isTraining = True):
        self.activation = prev_activation.reshape(prev_activation.shape[0],self.OutDim)
        return self.activation
    
    def backprop(self,dA):
        da_prev = dA.reshape(dA.shape[0],*self.PrevInput)
        return da_prev

    def update_parameters(self,lr,optimizer = LROptimizerType.NONE,iternum=0):
        return
    def dump(self):
        np.savetxt('c:/import/flatten'+str(self.OutDim)+'.csv')
        return
    def GetSize(self):
        size = self.OutDim
        return size

    def GetType(self):
        return "Flatten "
        
class AverageLayer(object):

    def __init__(self,prev_input):
        self.previnput = prev_input
        self.prevH, self.prevW,self.prevC = prev_input
        self.poolsize = 2
        self.outputSize = (math.floor(self.prevH/2),math.floor(self.prevW/2),self.prevC)
        self.dA = np.zeros(self.previnput)

    def forward(self,prev_activation,isTraining=True):
        
        self.samplesize = prev_activation.shape[0]
        self.output = np.zeros((self.samplesize,*self.outputSize))
        for sample in range(self.samplesize):
            for channel in range(0,self.prevC):
                for row in range(0,math.floor(self.prevH/2)):
                    for col in range(0,math.floor(self.prevW/2)):
                        self.output[sample,row,col,channel] = np.mean(prev_activation[sample,row*self.poolsize:row*self.poolsize+self.poolsize,col*self.poolsize:col*self.poolsize+self.poolsize,channel])
        return self.output

    def backprop(self,dA):
        samples,rows,cols,channels = dA.shape
        da_prev = np.zeros((samples,*self.previnput))
        for sample in range(samples):
            for channel in range(channels):
                for row in range(rows):
                    for col in range(cols):
                        da_prev[sample,row*self.poolsize,col*self.poolsize,channel] = dA[sample,row,col,channel]/4
                        da_prev[sample,row*self.poolsize+1,col*self.poolsize,channel] = dA[sample,row,col,channel]/4
                        da_prev[sample,row*self.poolsize,col*self.poolsize+1,channel] = dA[sample,row,col,channel]/4
                        da_prev[sample,row*self.poolsize+1,col*self.poolsize+1,channel] = dA[sample,row,col,channel]/4
        return da_prev
    
    def update_parameters(self,lr,optimizer = LROptimizerType.NONE,iternum=0):
        return
    def dump(self):
        return
    def GetSize(self):
        size = self.outputSize
        return size
    def GetType(self):
        return "Average Pooling "
def loadbackflat():
    da_prev = np.zeros((1,4,4,12))
    for counter in range(12):
        da_prev[0,:,:,counter] = np.loadtxt('c:/import/backflat'+str(counter)+'.csv',delimiter=',')
    return da_prev

class Model(object):
    """description of class"""
    def __init__(self,number_epochs=10, batch_size=1, stochastic=False,lr=0.1):
        self.layers = []
        self.epochs =  number_epochs
        self.batchsize = batch_size
        self.stochastic = stochastic
        self.lr = lr
        
        self.optimizer = LROptimizerType.NONE
  
    def AddDense(self,neurons,activationf,prevInput=None):
        if prevInput is not None:
            prev_a = prevInput
            self.x_inputs = prevInput
        else:
            prev_a = self.layers[-1].GetSize()
        layer = DenseLayer(neurons,prev_a,activationf)
        self.layers.append(layer)
        self.lastlayerAF = layer.activationf
        return

    def AddCNN(self,filters,filtersize,activationf,prevInput=None):
        if prevInput is not None:
            prev_a = prevInput
            self.x_inputs = prevInput
        else:
            prev_a = self.layers[-1].GetSize()
        layer = CNNLayer(prev_a,filters,filtersize,activationf)
        self.layers.append(layer)
        self.lastlayerAF = layer.activationf
        return

    def addAvgPool(self):
        prev_a = self.layers[-1].GetSize()
        layer = AverageLayer(prev_a)
        self.layers.append(layer)
        return

    def addFlatten(self,prevInput=None):
        if prevInput is not None:
            prev_a = prevInput
            self.x_inputs = prevInput
        else:
            prev_a = self.layers[-1].GetSize()
        layer = FlattenLayer(prev_a)
        self.layers.append(layer)
        return

    def ModelSummary(self):
        for layer in self.layers:
            print(layer.GetType(),layer.GetSize())

    def SetBatchNorm(self,Turnon):
        for layer in self.layers[:-1]:
            layer.isBatchNorm = Turnon
    def fit(self,X,Y):
        self.samples = X.shape[0]
        batches = math.floor(self.samples / self.batchsize)
        if batches * self.batchsize < self.samples:
            batches += 1

        for epoch in range(self.epochs):
            cost = 0
            iteration = 0
            start = timer()
            for batch in range(batches):
                startbatch = batch * self.batchsize
                endbatch = min((batch+1)*self.batchsize,self.samples)
                a_prev = X[startbatch:endbatch ]
                y_sample = Y[startbatch: endbatch]
                ## Forward Calculate
                for layer in self.layers:
                    a_prev = layer.forward(a_prev,isTraining=True)
                cost += loss(a_prev,y_sample,self.lastlayerAF)  

                # Backward Propagation

                da_prev = -(y_sample-a_prev)  # assume softmax

                for layer in reversed(self.layers):
                    da_prev = layer.backprop(da_prev).astype(float)
                    
                for layer in reversed(self.layers):
                    layer.update_parameters(self.lr,self.optimizer,iteration)
                print("epoch = " + str(epoch) + " batch = " + str(batch) + " loss = " + str(cost)) 
            end = timer()

            print('epoch ' + str(epoch) + ' Took ',end-start)

  
    def predict(self,X):
        a_prev = X
        for layer in self.layers:
            a_prev = layer.forward(a_prev,isTraining=False)
        return a_prev






