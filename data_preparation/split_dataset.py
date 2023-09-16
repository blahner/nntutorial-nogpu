import numpy as np

if __name__ == '__main__':
    #split the 60,000 training images into train and val sets and save
    data_path = "/mnt/g/nntutorial_copy/data/"
    train_split = 0.8
    val_split = 1 - train_split
    train_val_data = np.loadtxt(data_path + "mnist_train_val.csv", delimiter=",")
    split_indices = np.random.permutation(train_val_data.shape[0]) #random permutation vector the length of train/val images
    numTrainSamples = int(train_split * train_val_data.shape[0])
    train_split = split_indices[:numTrainSamples] #indices for training data
    val_split = split_indices[numTrainSamples:] #indices for validation data

    #save
    np.savetxt(data_path + "mnist_overfit.csv", train_val_data[split_indices[:100],:], fmt='%d', delimiter=",")
    np.savetxt(data_path + "mnist_train.csv", train_val_data[train_split,:], fmt='%d', delimiter=",")
    np.savetxt(data_path + "mnist_val.csv", train_val_data[val_split,:], fmt='%d', delimiter=",")


