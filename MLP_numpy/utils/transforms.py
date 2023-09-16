import numpy as np

class Normalize(object):
    def __init__(self, max_val = 255, mean = 0.1307, std = 0.3081):
        #normalize images within a range, mean centered at 0 and standard deviation of 1. Values specific to MNIST
        self.max_val = max_val
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]
        img = img/self.max_val #fix range
        img = (img - self.mean)/self.std
        return {"image": img, "label": sample["label"]}

class InverseNormalize(object):
    #useful for visualization
    def __init__(self, mean = 0.1307, std = 0.3081):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]
        img = img*self.std + self.mean
        return {"image": img, "label": sample["label"]}
