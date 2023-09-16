import os

import numpy as np

class mnist_numpy_dataset(object):
    def __init__(self, root_dir, phase='train', transforms=None):
        """
        Initialize the dataset instance of the custom "mnist_numpy_dataset" class. The 
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

    def getitem(self, idx):
        """
        Method to tell the dataset how to retrieve an image and label.
        This method mimics torch's "__getitem__" method that is usually inherited when you
        use the pytorch framework. Images get transformed and labels get turned into one hot
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
        img = self.data_file[idx,1:]
        label = self.data_file[idx,0]
        label_one_hot = np.zeros(10)
        label_one_hot[int(label)] = 1.
        sample = {'image': img, 'label': label_one_hot}
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample