import numpy as np

class mnist_numpy_dataloader(object):
    def __init__(self, dataset, batch_sz = 4, shuffle = True):
        """
        This initializes our mnist_numpy_datalaoder class. Initialilze some class variables
        to define how we are going to feed in our MNIST images and labels to the network.
        
        Parameters
        ----------
        dataset: "mnist_numpy_dataset" class object
            Instance of our custom "mnist_numpy_dataset" class.
        batch_sz: int, optional
            defines how many images are in each batch. Defaults to 4
        shuffle: bool, optional
            If true, shuffle the image order in the dataloader. If false, do not shuffle. Defaults to True.
        """
        self.dataset = dataset
        self.batch_sz = batch_sz
        self.shuffle = shuffle
        self.batch_start = np.arange(0,len(self.dataset), self.batch_sz) #start indices in the batch
        self.batch_stop = np.append(self.batch_start[1:], len(self.dataset)) #end indices in the batch
        self.n_batches = len(self.batch_start)
        iter(self)

    def load_batch_indices(self):
        """
        Precomputes the image indices in each of the batches to load in the future.
        """
        if self.shuffle:
            p = np.random.permutation(len(self.dataset))
        else:
            p = np.array(range(len(self.dataset)))

        batched_indices = [] #list of indices in each batch 
        for i in range(self.n_batches):
            batched_indices.append(p[self.batch_start[i]:self.batch_stop[i]])

        self.batched_indices = batched_indices
        return self

    def __iter__(self):
        """
        Enables us to iterate over the dataloader.
        """
        self.n = 0 #we start at a batch index of zero
        return self

    def __next__(self):
        """
        Defines what happens at each iteration. At each iteration, we return the images and corresponding labels at 
        the current batch and increase the batch index to prepare the loading of the next batch at the next function call. 
        """
        if self.n < self.n_batches: #only loads barch_sz number of images and labels for the corresponding batch on its for loop
            if self.n == 0: #if this is the first call, shuffle the images
                self.load_batch_indices() #just get the indices for all the batches. Loads the actual images and labels separately
            curr_batch = self.batched_indices[self.n]
            batch_img = []
            batch_label = []
            for b in curr_batch:
                sample = self.dataset.getitem(b)
                batch_img.append(sample["image"])
                batch_label.append(sample["label"])
            self.n += 1 #increments the counter to the next batch upon calling "next" again with the for loop
            return np.array(batch_img), np.array(batch_label)
        else:
            raise StopIteration #traverse all batches. This quits the external for loop
    
    def __len__(self):
        """
        We define the length of the dataloader as the number of batches.
        """
        return self.n_batches