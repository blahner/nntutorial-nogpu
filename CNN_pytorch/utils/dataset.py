import numpy as np
import os
from torch.utils.data import Dataset

class cnn_pytorch_dataset(Dataset): #with pytorch we inherit the Dataset class
    def __init__(self, root_dir, phase='train', transforms=None):
        """
        Initialize the dataset instance of the custom "mnist_pytorch_dataset" class. The 
        dataset class is responsible for naively retrieving and transforming images and labels
        from the MNIST dataset. It will be input to a dataloader class that will more specifically
        define batch sizes, shuffling etc. 
        """
        self.root_dir = root_dir
        self.data_file = np.loadtxt(os.path.join(self.root_dir, "mnist_" + phase + ".csv"), delimiter=",")
        self.transforms = transforms

    def __len__(self):
        """
        We tell python to return the length of the datafile as the length of the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Method to tell the dataset how to retrieve an image and label.
        Images get transformed and labels get turned into one hot
        vectors.

        Parameters
        ----------
        idx : int
            The index into the dataset of the image and label you want to retrieve

        Returns
        -------
        sample: dictionary
            A dictionary with keys "image" and "label" corresponding to transformed images
            and one hot labels.

        """
        img = self.data_file[idx,1:].reshape(1,28,28) #reshape data from a 1d vector into a 3d matrix (i.e. image of size 1 channel, 28 height, 28 width)
        label = self.data_file[idx,0]
        label_one_hot = np.zeros(10)
        label_one_hot[int(label)] = 1.
        sample = {'image': img, 'label': label_one_hot}
        if self.transforms:
            sample = self.transforms(sample)

        return sample