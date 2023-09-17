import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .transforms import InverseNormalize


def factor(num):
    """
    Factor a number to x1*x2 where x1 and x2 are as close to being
    squares of input NUM as possible. Helps for plotting various sizes of
    layers and weights.

    Parameters
    ----------
    num : int
        number that you want factored

    Returns
    -------
    factors_list : list of tuples
        Each index of FACTORS_LIST is a tuple of (x1, x2) factors of NUM

    """
    factors_list = []
    max_root = int(np.ceil(np.sqrt(num)))
    for x1 in np.arange(1, max_root+1):
        if num % x1 == 0: 
            x2 = int(num/x1)
            factors_list.append((x1, x2))
    return factors_list

def plot_sample_imgs(dataset, save_path=None):
    """
    Plots sample images from a dataset for visualization purposes and saves the plots.
    
    Parameters:
    -----------
    dataset: "mnist_numpy_dataset" class object
        dataset object of the MNIST dataset
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.
    
    Returns
    -------
    None.
    """
    transform_inverse = InverseNormalize() #define an inverse normalize object. Since MNIST is grayscale, inverse normalizing doesn't make a big difference for visualizing. This is useful for natural images though
    fig, ax = plt.subplots(3,3)
    fig.suptitle("Sample Images from Training Set")
    for r in range(3):
        for c in range(3):
            idx = 3*r + c
            sample = dataset[idx] #with the pytorch dataset class we can just index into the dataset. It automatically calls the __getitem__ method
            img = sample["image"].squeeze()
            label = sample["label"]
            
            ax[r,c].imshow(img, cmap="Greys")
            label, = np.where(label == 1)
            ax[r,c].set_title("Label: " + str(label[0]))
            ax[r,c].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.clf()

def visualize_convweights(model, cnnlayer, save_path=None):
    """
    Visualize layer weights of the network in one plot. If the network is properly
    trained, we should be able to see some patterns in the weights. 
    Plot the kernel (k_h x k_w) for each input channel and output channel 

    Parameters
    ----------
    model : nn.Module class
        keys are the weights and biases of each layer
    cnnlayer : int
        Use this to index which layer's weights you want to visualize
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.

    Returns
    -------
    None.

    """
    fs = 20 #fontsize for plots
    #weights are shape: (out_channels, groups/in_channels, kernel_size[0], kernel_size[1])
    
    weight_values = eval('model.CNN' + str(cnnlayer) + '[0].weight.detach().cpu().numpy()') # shape [out_ch, in_ch, k_h, k_w]
    #for out_channel in weight_values.shape
    out_ch, in_ch, k_h, k_w = weight_values.shape
    #rows_subplot = int(np.floor(np.sqrt(out_ch)))
    #cols_subplot = int(np.ceil(out_ch/rows_subplot))
    #fig, ax = plt.subplots(rows_subplot, cols_subplot)
    fig, ax = plt.subplots(out_ch, in_ch, squeeze=False)
    fig.suptitle("Weight Visualization of Conv Layer " + str(cnnlayer), fontsize = fs)
    #factors_list = factor(weight_values.shape[1])
    #best_factor = factors_list[-1] #factors x1 and x2 that are closest to squares
    #r = best_factor[0]
    #c = best_factor[1]
    #print("Weight plots are " + str(r) + " pixels by " + str(c) + " pixels.")
    print("Weight plots are " + str(k_h) + " pixels by " + str(k_w) + " pixels.")
    for out_ch_idx in range(out_ch):
        weights_out = weight_values[out_ch_idx,:,:,:]
        for in_ch_idx in range(in_ch):
            weights_in = weights_out[in_ch_idx,:,:]
            ax[out_ch_idx, in_ch_idx].imshow(weights_in)
            ax[out_ch_idx, in_ch_idx].axis(False)
    if save_path:
        plt.savefig(save_path + "_layer-" + str(cnnlayer) + ".png")
    plt.show()
    plt.clf()

def visualize_activations(activation_values, img_idx=0, save_path=None):
    """
    Visualize the activations of an image as it gets passed through the network.
    Note how the activations change depending on whether the network is trained
    or not. 

    Parameters
    ----------
    activation_values : Dictionary
        Dictionary of activation values for each image at each layer in the model.
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.

    Returns
    -------
    None.

    """
    for key in activation_values.keys():
        act = activation_values[key][img_idx].detach().cpu().numpy().squeeze()
        shape = act.shape[0]
        if shape == 10: #output layer
            best_factor = (1,10)
        else:
            factors_list = factor(shape)
            best_factor = factors_list[-1] #factors x1 and x2 that are closest to squares
        r = best_factor[0]
        c = best_factor[1]
        act_reshape = act.reshape((r,c))
        plt.imshow(act_reshape)
        if shape == 10:
            for j in np.arange(10):
                plt.text(j, 0, round(act_reshape[0,j],2), ha="center", va="center", color="k")
            plt.xticks(ticks = np.arange(10), labels = np.arange(10))
            plt.yticks(ticks=[])
            plt.title("MNIST Digit Confidence")
        else:
            plt.title("Reshaped Activation of " + key)        
        if save_path:
            plt.savefig(save_path + "_" + key + ".png")
        plt.show()
        plt.clf()
