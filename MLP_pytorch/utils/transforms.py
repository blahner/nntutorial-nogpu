import torch

class Normalize(object):
    """
    normalize images within a range, mean centered at 0 and 
    standard deviation of 1. Max_val, mean, and std values specific to MNIST.
    """
    def __init__(self, max_val = 255, mean = 0.1307, std = 0.3081):
        """
        Initialize the instance parameters. Specific to MNIST
        """
        self.max_val = max_val
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Defines how inputs are processed in this class.

        Parameters
        ----------
        sample : dictionary
            contains "image" and "label" keys containing the image and label

        Returns
        ----------
        dictionary
            keys of "image" and "label" of the transformed image and label
        """
        img = sample["image"]
        img = img/self.max_val #fix range
        img = (img - self.mean)/self.std
        return {"image": img, "label": sample["label"]}

class ToTensor1D(object): #works for 1D arrays
    """"
    Inputs to networks should be tensors for pytorch to keep track of the gradients
    """
    def __init__(self):
        return None
    
    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        img_tensor = torch.Tensor(img)
        label_tensor = torch.Tensor(label)
        return {"image": img_tensor, "label": label_tensor}

class InverseNormalize(object):
    """
    Inverse normalization is useful for visualization. Simply undos
    the normalization procedure.
    """
    def __init__(self, mean = 0.1307, std = 0.3081):
        """
        Initialize the instance parameters. Specific to MNIST
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Defines how inputs are processed in this class.

        Parameters
        ----------
        sample : dictionary
            contains "image" and "label" keys containing the image and label

        Returns
        ----------
        dictionary
            keys of "image" and "label" of the transformed image and label
        """
        img = sample["image"]
        img = img*self.std + self.mean
        return {"image": img, "label": sample["label"]}
