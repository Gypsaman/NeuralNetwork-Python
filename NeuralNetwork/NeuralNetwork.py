import sys

import NNModel as nn
import os
import cv2
from sklearn.utils import shuffle
from ActivationType import ActivationType
import numpy as np

def get_Densedata(dir):
    images = os.listdir(dir)
    numimages = len(images)
    X = np.empty((numimages,28,28),dtype='float64')
    Y = np.zeros((numimages,10))

    i = 0
    for image in shuffle(images):
        digit = int(image[0])
        Y[i,digit] = 1.0
        im = cv2.imread("{0}/{1}".format(dir,image),0)
        X[i,:,:] = im/255.0
        i += 1
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    return X,Y

def get_CNNdata(dir):
    images = os.listdir(dir)
    numimages = len(images)
    X = np.empty((numimages,28,28),dtype='float64')
    Y = np.zeros((numimages,10))

    i = 0
    for image in images:
        digit = int(image[0])
        Y[i,digit] = 1.0
        im = cv2.imread("{0}/{1}".format(dir,image),0)
        X[i] = im/255.0
        
        i += 1
    return X.reshape(*X.shape,1),Y

def runmodel(batchsize,trainX,trainY,testX,testY,epoch,hiddenlayer,batchnorm=False):

    layers=[(hiddenlayer,ActivationType.SIGMOID),(10,ActivationType.SIGMOID)]
    model = nn.Model(number_epochs=epoch,batch_size=batchsize,stochastic=False,lr=0.1)
    size,activationf = layers[0]
    model.AddDense(size,activationf,(784))
    for size,activationf in layers[1:]:
        model.AddDense(size,activationf)
    model.SetBatchNorm(batchnorm)
    model.fit(trainX,trainY)
    y_predict = model.predict(testX)

    matches = 0
    for i in range(testX.shape[1]):
        index = y_predict[:,i].argmax(axis=0)
        if testY[index,i] == 1:
            matches += 1
    return matches
def resultplot(results):
    import matplotlib.pyplot as plt

    epochs = [25,50,100,150]
    fig = plt.figure()
    sgd = fig.add_subplot(1,2,1)
    sgd.set_title('SGD')
    sgd.set_xlabel("Hidden Layers")
    sgd.set_ylabel("% Predicted")
    for i in range(4):
        sgd.plot([25,50,100,150],results[0,i,:]/100,label='epoch-'+str(epochs[i]))
    sgd.legend()

    bat = fig.add_subplot(1,2,2)
    bat.set_title('Mini Batch')
    for i in range(4):
        bat.plot([25,50,100,150],results[1,i,:]/100,label='epoch-'+str(epochs[i]))
    bat.set_yticklabels([])
    bat.set_xlabel("Hidden Layers")
    bat.legend()
    plt.show()
    plt.savefig('c:\import\image')

def testepochs():
    hiddenlayers = [25,50,100,150]
    epochs = [25,50,100,150]
    results = np.zeros((2,4,4))
    epochnum = 0
    for epoch in epochs:
        hiddenlayernum = 0
        for hiddenlayer in hiddenlayers:
            matches = runmodel(20,trainX,trainY,testX,testY,epoch,hiddenlayer)
            results[0,epochnum,hiddenlayernum] = matches
            matches = runmodel(10,trainX,trainY,testX,testY,epoch,hiddenlayer)
            results[1,epochnum,hiddenlayernum] = matches
            hiddenlayernum += 1
        epochnum += 1
    resultplot(results)
def testbatchnorm():
    np.random.seed(1105)
    matchesoff = runmodel(20,trainX,trainY,testX,testY,100,100,False)
    matcheson = runmodel(20,trainX,trainY,testX,testY,100,100,True)
    print("BatchNorm on ",matcheson)
    print("BatchNorm off ",matchesoff)
    return
def test():
    mat = np.array([[0.855340935339027, 0.659243669462703, 2.19756999966326, -0.0613551872500569, 0.281570788377319 ],[-0.297828541793396, 0.975843080957277, -1.59162885920429, 0.731631882312554, 0.734783528221844],[1.44826903774082, -0.112689934246199, 0.585660133687248, -1.84524375964317, 1.28698493110012 ],[0.386107801332689, 1.18402313522654, 1.11877286819708, 1.37491998510238, -0.390899676365182 ],[-0.368185780281517, 0.849433398660155, 0.389727782555085, 0.536255457553567, 0.383417412921244]])
    kern = np.array([[1,2,1],[0,0,0],[1,2,1]])
    res = nn.fullconvolve(mat,kern)
    np.savetxt('c:/import/mat.csv',mat,delimiter=',')
    np.savetxt('c:/import/kern.csv',kern,delimiter=',')
    np.savetxt('c:/import/res.csv',res,delimiter=',')
    return
def main():
    #test()
    np.random.seed(11051966)
    trainX,trainY = get_CNNdata('c:/users/cgarcia/CPEG586/minst/Training')
    testX,testY = get_CNNdata('c:/users/cgarcia/CPEG586/minst/Test')

    model = nn.Model(number_epochs=30,batch_size=100,lr=0.1 )
    model.AddCNN(6,5,ActivationType.RELU,prevInput=(28,28,1))
    model.addAvgPool()
    model.AddCNN(12,5,ActivationType.RELU)
    model.addAvgPool()
    model.addFlatten()
    model.AddDense(50,ActivationType.RELU)
    model.AddDense(10,ActivationType.SOFTMAX)

    model.ModelSummary()

    model.fit(trainX,trainY)
    
    y_predict = model.predict(testX)

    matches = 0
    for i in range(testX.shape[0]):
        index = y_predict[i,:].argmax(axis=0)
        if testY[i,index] == 1:
            matches += 1

    return
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))
