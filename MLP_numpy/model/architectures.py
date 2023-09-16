import numpy as np

class MLP_numpy(object):
    def __init__(self, layer_sz):
        """
        This initializes our MLP_NUMPY class. Initialilze some class variables
        to keep track of number of layers, activations, activations 
        after non-linearity, and parameters. This class is our network architecture 
        definition. It is fundamentally composed of the number of layers, types of layers,
        value of weights and biases, and nonlinear functions. Together, the architecture mathematically 
        describes how inputs (our MNIST images) are manipulated by the weights through different layers,
        and with the cost function, describes how these network weights are updated to increase the accuracy.
        
        Parameters
        ----------
        layer_sz : list in int
            Each element in index "i" defines the size (number of nodes) for
            layer "i" in our MLP network

        """
        #assertions for input and output sizes. We don't care about the number or sizes of hidden layers
        assert(layer_sz[0] == 784) #with MNIST, input size must be 784
        assert(layer_sz[-1] == 10) #with MNIST, output size must be 10
        
        self.layer_sz = layer_sz
        self.numLayers = len(self.layer_sz)-1 #number of layers is number of trainable layers i.e. all layers except for input
        self.a = {} # keeps track of the "a" term in the math eqquations for each layer
        self.z = {} # keeps track of the "z" term in the math equations for each layer
        self.param = {} # keeps track of the weights and biases for each layer
        for i in range(1, self.numLayers + 1): #the convention I use here is layer 0 contains the full image input. Thus, there are no weight and bias values in layer 0
            #He initialization
            #self.param["Wlayer" + str(i)] = np.random.randn(self.layer_sz[i], self.layer_sz[i-1]) * np.sqrt(2/self.layer_sz[i-1]) #initialize weights
            #self.param["Blayer" + str(i)] = np.zeros((self.layer_sz[i],1)) #initialize biases
            #initialize weights with normal distribution
            #self.param["Wlayer" + str(i)] = np.random.randn(self.layer_sz[i], self.layer_sz[i-1])
            #self.param["Blayer" + str(i)] = np.zeros((self.layer_sz[i],1)) #initialize biases with zeros
            #initialize weights with uniform distribution
            #self.param["Wlayer" + str(i)] = np.random.uniform(size = (self.layer_sz[i], self.layer_sz[i-1]))
            #self.param["Blayer" + str(i)] = np.zeros((self.layer_sz[i],1)) #initialize biases with zeros
            #initialize weights with pytorch default
            stdv = 1./np.sqrt(self.layer_sz[i-1])
            self.param["Wlayer" + str(i)] = np.random.uniform(low = -stdv, high = stdv, size = (self.layer_sz[i], self.layer_sz[i-1]))
            self.param["Blayer" + str(i)] = np.zeros((self.layer_sz[i],1)) #initialize biases with zeros

    def sigmoid_forward(self, z):
        """
        Sigmoid function in the forward direction. This is our nonlinearity in the network.

        Parameters
        ----------
        z : numpy array
            This is our activation before nonlinearity

        Returns
        -------
        a : numpy array
            This is our activation after nonlinearity

        """
        a = 1/(1 + np.exp(-z))
        return a
    
    def sigmoid_backward(self, a):
        """
        Sigmoid function in the backward direction. This is the derivative
        of the sigmoid function. We use this to propogate the
        gradient backwards through the network.

        Parameters
        ----------
        a : numpy array
            This is our activation after the nonlinearity

        Returns
        -------
        z : numpy array
            This is our activation before nonlinearity

        """
        z = a*(1 - a)
        return z
    
    def softmax(self, z):
        """
        Softmax function in the forward direction. This is our nonlinearity in the network.

        Parameters
        ----------
        z : numpy array
            This is our activation before nonlinearity
        
        Returns
        ----------
        a : numpy array
            This is our activation after nonlinearity 
        """
        exp=np.exp(z-z.max())    
        a = exp/np.sum(exp,axis=0)
        return a
    
    def softmax_backward(self, a):
        """
        Softmax function in the backward direction. This is the derivative
        of the softmax function. We use this to propogate the
        gradient backwards through the network.

        Parameters
        ----------
        a : numpy array
            This is our activation after the nonlinearity

        Returns
        -------
        z : numpy array
            This is our activation before nonlinearity

        """
        exp = np.exp(a-a.max())
        z = exp/np.sum(exp,axis=0)*(1-exp/np.sum(exp,axis=0))
        return z
    
    def forward_single_layer(self, x, layer):
        """
        Define the forward direction for just a single layer. 

        Parameters
        ----------
        x : numpy array
            Input to the layer
        layer : int
            Which layer is the forward direction is being computed for. Used
            to index the appropriate parameters

        Returns
        -------
        z : numpy array
            This is our activation before nonlinearity
        a : numpy array
            This is our activation after the nonlinearity

        """
        z = np.dot(self.param["Wlayer" + str(layer)], x) + self.param["Blayer" + str(layer)]
        a = self.sigmoid_forward(z)
        return z, a
    
    def forward_all_layers(self, input_vec):
        """
        Compute the forward direction for all the layers in our network. Use
        our function "forward_single_layer" to help put it all together.

        Parameters
        ----------
        input_vec : numpy array
            input matrix

        Returns
        -------
        out_vec : numpy array
            output of the model before loss function

        """
        input_vec = input_vec.T #a bit of matrix gymnastics to make the shapes work
        self.a["layer0"] = input_vec #define layer 0 as input. Not trainable i.e. no weights or biases in layer0
        for i in range(1, self.numLayers+1): #loop through layers
            z, a = self.forward_single_layer(input_vec, i) #perform the forward pass on this layer
            self.a["layer" + str(i)] = a #save 'a' for backprop
            input_vec = a #the input to the next layer is the output of the previous layer
        out_vec = a #the last output activation vector is the output
        return out_vec
        
    def backward_single_layer(self, dc_da, layer, learning_rate):
        """
        Define the backward direction for just a single layer.

        Parameters
        ----------
        dc_da : numpy array
            partial derivative of cost (c) with respect to activation (a) 
        layer : int
            which layer we are performing the backward pass on
        learning_rate : float
            how much to update the weights and biases of the network

        Returns
        -------
        numpy array
            the 'dc_da' that will be propogated to the next layer of the model
            to continue the backward pass through all layers.

        """
        da_dz = self.sigmoid_backward(self.a["layer" + str(layer)])
        dc_dz = np.multiply(dc_da, da_dz)
        
        dz_dw = self.a["layer" + str(layer-1)].T
        dz_da = self.param["Wlayer" + str(layer)].T
        dz_db = 1
        
        dc_da_next = np.dot(dz_da, dc_dz) #outer product. Use this as dc_da input for next layer 
        dc_dw = np.dot(dc_dz, dz_dw) #what we've been waiting for!
        dc_db = np.sum(np.multiply(dc_dz, dz_db), axis=1, keepdims=True) #what we've been waiting for!
        
        #update the weights and biases from the gradient
        self.param["Wlayer" + str(layer)] -= learning_rate * dc_dw
        self.param["Blayer" + str(layer)] -= learning_rate * dc_db
        
        return dc_da_next #return the next dc_da matrix to propagate the gradient to the next layer
        
    def backward_all_layers(self, dc_da, learning_rate):
        """
        Compute the backward direction for all the layers in our network. Use
        our function "backward_single_layer" to help put it all together.

        Parameters
        ----------
        prediction : numpy array
            model predictions. shape (num_classes, num_images)
        target : numpy array
            one-hot image labels. shape (num_classes, num_images)
        learning_rate : float
            how much to update the weights and biases of the network

        Returns
        -------
        None.

        """
        dc_da_curr = dc_da
        for i in range(self.numLayers, 0, -1):
            dc_da = self.backward_single_layer(dc_da_curr, i, learning_rate)
            dc_da_curr = dc_da 
            
    def get_layer_sz(self):
        """
        Simply returns "layer_sz"

        Returns
        -------
        layer_sz_copy : list of int
            list of the layer sizes in the models, where each element in the list
            is a layer index and the value is the size of that layer.

        """
        layer_sz_copy = self.layer_sz.copy()
        return layer_sz_copy
    
    def get_parameters(self):
        """
        Simply grabs the trainable parameters of the network. Useful to visualize
        the weights that we learned.

        Returns
        -------
        parameters_copy : dict
            keys are the weights and biases of each layer

        """
        parameters_copy = self.param.copy() #careful of mutability
        return parameters_copy
    
    def get_activations(self):
        """
        Gets the activations of the network. Useful to visualize how our input
        image was transformed at different layers in the network

        Returns
        -------
        activations_copy : dict
            keys are the activations of the images in each layer of the network.
            shape in each layer is (layer_size, num_images)

        """
        activations_copy = self.a.copy() #careful of mutability
        return activations_copy
    
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
        #get number of trainable parameters
        tw = 0 #initialize number of trainable weights
        tb = 0 #initialize number of trainable biases
        for i in range(0, self.numLayers):
            tw += (self.layer_sz[i]*self.layer_sz[i+1])
            tb += self.layer_sz[i+1]
        return int(tw+tb)
            
    def __str__(self):
        """
        Defines how we want the "print" function to work on our model
        """
        #I make the assumption that this MLP model uses a linear + sigmoid layer
        message = "This Multi-Layer Perceptron has an input layer, " + str(self.numLayers-1) + " hidden layers, " +\
            "and an output layer. It has a total of "\
            + str(self.count_parameters()) + " trainable parameters, including biases."
        return message
