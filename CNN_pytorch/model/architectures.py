import torch.nn as nn

class CNN_pytorch(nn.Module):
    """
    Input: (N, C_in, H_in, W_in)
    Output: (N, C_out, H_out, W_out), where
    
    H_out =[(H_in + 2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0]] + 1
    W_out =[(W_in + 2×padding[1]−dilation[1]×(kernel_size[1]−1)−1)/stride[1]] + 1
    """
    def __init__(self):
        super(CNN_pytorch, self).__init__()
        """
        This initializes our CNN_pytorch class. Initialilzes a network with two convolutional layers and one fully
        connected layer. The nonlinearity is a ReLU. We use max pooling to reduce the dimensionality in the layers.
        The underlying principles between convolutional layers and linear (fully connected) layers are the same, even
        though convolutional layers look very different. They both are linear combinations of the weights and inputs. 
        """
        self.CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2 
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # fully connected layer, output 10 classes
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10)
        )
        
        #now let's try a fancier way of initializing weights
        nn.init.kaiming_uniform_(self.CNN[0].weight,mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.CNN[3].weight,mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        """
        Define the forward direction for the network. Look how self-contained this code is
        by using pytorch.

        Parameters
        ----------
        x : numpy array
            Input to the layer
        Returns
        ----------
        output : array
            Output of the entire network.
        x : array
            Output of the convolutional layers. Useful for visualization
        """
        x = self.CNN(x)
        x = x.view(x.size(0), -1)       
        output = self.fc(x)
        return output, x    # return x for visualization

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)