import numpy as np
import activations as act
class Layer(object):
    """description of class"""
    def __init__(self, nuerons, inputs, activationf='Relu'):
        self.nodes = neurons
        self.inputs = inputs
        self.weights = np.random.random((neurons,inputs))
        self.biases = np.zeros((neurons,1))
        self.delta =  np.zeros((neurons,1))
        self.s = np.zeros((neurons,1))
        self.activationf = activationf
        self.activation = np.zeros((neurons,1))
    
    def forward(self,inputs):
        assert inputs.shape == (inputs,1)
        s = np.matmul(np.transpose(weights),inputs)
        if activationf == 'Relu':
            self.activation = act.relu(s)
        else:
            self.activation = act.sigmoid(s)
