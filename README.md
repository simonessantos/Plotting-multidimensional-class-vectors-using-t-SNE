# Plotting-multidimensional-class-vectors-using-t-SNE

[![Python3](https://img.shields.io/badge/python-3.6-green)](https://www.python.org/download/releases/3.0/)

# PCA 
![N|Solid](https://github.com/henriqueburis/Plotting-multidimensional-class-vectors-using-t-SNE/blob/main/figure/cifar100-3download%20(2).png?raw=true)
# t-SNE 
![N|Solid](https://github.com/henriqueburis/Plotting-multidimensional-class-vectors-using-t-SNE/blob/main/figure/cifar100-2download%20(2).png?raw=true)
# Digits tsne generated cluster
![N|Solid](https://github.com/henriqueburis/Plotting-multidimensional-class-vectors-using-t-SNE/blob/main/figure/cifar100download%20(2).png?raw=true)

## Installation
- from sklearn.datasets import load_files
- from keras.preprocessing.image import img_to_array
- from sklearn.cluster import KMeans
- from sklearn.preprocessing import LabelEncoder
- from sklearn import preprocessing
- from sklearn import decomposition
- from sklearn.manifold import TSNE
- import matplotlib.pyplot as plt
- import seaborn as sns
- import matplotlib.patheffects as PathEffects
- import numpy as np
- import cv2
- import argparse

## Using the t-SNE
you can now run the python scrypt with the following command:

```sh
python3 main.py  --dataroot ${'path file str'}
```

