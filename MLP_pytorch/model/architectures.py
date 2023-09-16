#notice how we don't even need to import numpy this time
import torch.nn as nn

class MLP_pytorch(nn.Module): #with pytorch we inherit the nn.Module class. So much easier!
    def __init__(self, layer_sz):
        """
        This initializes our MLP_pytorch class. Initialilzes our network based on the layer size we want.
        We define the specific layers using pytorch though, making this code much simpler than the MLP_numpy class.
        This tutorial uses a lot of space just to accommodate different network layer combinations, so imaging this
        code can be even shorter if we know beforehand our desired layer sizes.
        
        Parameters
        ----------
        layer_sz : list in int
            Each element in index "i" defines the size (number of nodes) for
            layer "i" in our MLP network

        """
        super(MLP_pytorch, self).__init__() #also nn.Module.__init__(self) #inherit the attributes of nn.Module super/parent class
        assert(layer_sz[0] == 784) #with MNIST, input size must be 784
        assert(layer_sz[-1] == 10) #with MNIST, output size must be 10
        self.numLayers = len(layer_sz)-1 #number of layers is number of trainable layers i.e. all layers except for input

        layers = []
        for i in range(0,self.numLayers-1):
            layer = nn.Linear(layer_sz[i], layer_sz[i+1])
            #initialize weights
            #nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='sigmoid')
            nn.init.xavier_normal_(layer.weight)
            
            layers.append(layer)
            layers.append(nn.Sigmoid())
        
        layers.append(nn.Linear(layer_sz[-2], layer_sz[-1]))
        layers.append(nn.Sigmoid())  #layers.append(nn.Softmax(dim=1))
        
        #"nn.Sequential" essentially feeds the outputs of one layer as inputs to the next, making our network deffinition
        #very easy
        self.MLP = nn.Sequential(
            *layers #unpacking the "layers" variable
            )
        
    def forward(self, x): #forward gets called automatically when given an input
        """
        Define the forward direction for the network. Look how self-contained this code is
        by using pytorch.

        Parameters
        ----------
        x : array
            Input to the layer
        
        Returns
        ----------
        x : array
            Output of the layer used as input to the next.
        """
        x = self.MLP(x)
        return x
    
    def count_parameters(self):
        """
        Counts how many trainable parameters are in our network. Networks are
        often defined partly by how many trainable parameters they contain, as this 
        often is a good indicator of the networks complexity.

        Returns
        -------
        int
            number of trainable weights and biases in the model

        """
        return sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)
    
    def __str__(self):
        """
        Defines how we want the "print" function to work on our model
        """
        message = "This Multi-Layer Perceptron has an input layer, " + str(self.numLayers-1) + " hidden layers, " +\
            "and an output layer. It has a total of "\
            + str(self.count_parameters()) + " tranable parameters, including biases."
        return message