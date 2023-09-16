import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    data_path = "/mnt/g/nntutorial_copy/data/"
    
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    #view example test data output
    test_data[:10]
    
    #normalize all values between (0, 1]
    print("Max test_data value:", np.max(test_data))
    print("Min test_data value:", np.min(test_data))
    
    test_imgs = test_data[:,1:]/np.max(test_data[:,1:]) + 0.001
    train_imgs = train_data[:,1:]/np.max(train_data[:,1:]) + 0.001
    
    test_labels = test_data[:,:1]
    train_labels = train_data[:,:1]
    
    #preallocate one-hot encoding of labels
    test_labels_one_hot = np.zeros((len(test_labels), 10))
    train_labels_one_hot = np.zeros((len(train_labels), 10))
    
    for count, label in enumerate(test_labels):
        test_labels_one_hot[count, int(label)] = 1.
        
    for count, label in enumerate(train_labels):
        train_labels_one_hot[count, int(label)] = 1.  
    
    #show sample images and labels
    for i in range(10):
        img = train_imgs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        label, = np.where(train_labels_one_hot[i,:] == 1)
        plt.title("Number: " + str(label[0]))
        plt.show()

    #Dump mnist dataset and labels into pickle for faster loading
    with open("mnistnumpy/data/pickled_mnist.pkl", "bw") as fh:
        data = (train_imgs,\
                test_imgs,\
                train_labels,\
                test_labels,\
                train_labels_one_hot,\
                test_labels_one_hot)
        pickle.dump(data, fh)