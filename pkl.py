import numpy as np
import _pickle as cPickle


path = "/home/leech/Downloads/data/segmentation_results/CAMERA25/results_val_00000_0000.pkl"
with open(path, 'rb') as f:
    nocs = cPickle.load(f)

print(nocs.keys())