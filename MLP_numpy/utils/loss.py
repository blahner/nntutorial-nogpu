import numpy as np

class MSELoss(object):
    def __init__(self):
        """
        Initialize the Mean Squared Error cost function. The cost function
        is a measure of how good (or bad) our prediction was, telling 
        us how to update our network.
        """
        return None

    def __call__(self, predicted, target):
        """
        Defines what happens when we give input to an instance of this class.

        Parameters
        ----------
        predicted : numpy array
            the output activations of each of the 10 classes (digits 0-9) 
            of the model for each image. shape (num_classes, num_images)
        target : numpy array
            one-hot matrix of target labels for each image. shape (num_images, num_classes)

        Returns
        -------
        float
            The mean squared error loss of the model predictions

        """
        l = (target-predicted.T) ** 2
        return l.mean()

    def derivative(self, predicted, target):
        """
        Derivative of the Mean Squared Error Loss. The derivative is used for 
        the calculation of gradients for backpropogation.

        Parameters
        ----------
        predicted : numpy array
            the output activations of each of the 10 classes (digits 0-9) 
            of the model for each image. shape (num_classes, num_images)
        target : numpy array
            one-hot matrix of target labels for each image. shape (num_images, num_classes)

        Returns
        -------
        float
            The derivative of the mean squared error loss of the model predictions
        """

        m = target.shape[1]
        dc_da_curr = -2*(target.T - predicted)/m #MSE for the batch
        return dc_da_curr