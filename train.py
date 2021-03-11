from sklearn.datasets import load_files
from keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
import numpy as np
import cv2
import argparse


#parser = argparse.ArgumentParser(description='TSN')
#parser.add_argument('--dataroot',type=str,required=True,
 #                   help='path of data')
#args = parser.parse_args()

pathfile = "/content/CIFAR10-MIXUP/CIFAR10-00199-MSGAN+ECE_filtered_LOW_ENTROPY_1e-2+TRA+DA"
data = load_files(pathfile,random_state=0, load_content=True)

raw_data = []
raw_labels = list()

for idx2 in range(len(data['filenames'])):
  image = cv2.imread(data['filenames'][idx2])
  image = img_to_array(image)
  image = cv2.resize(image, (28, 28)).flatten()
  raw_data.append(image)

raw_data=np.array(raw_data)
raw_labels = data['target']

print(raw_data.shape)
print(np.max(raw_data))
print("number of Class: ",len(data['target_names']))
print(np.unique(data['target_names']))
print(np.unique(raw_labels))


PMCAT_List = list(data['target_names'])
kmeans_cluster = len(data['target_names'])
nsamples = 5000
Label_mcat = raw_labels[:nsamples]
data = raw_data[:nsamples]
labels = raw_labels[:nsamples]

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 18))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(18):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def scatter_2(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 11))

    # We create a scatter plot.
    f = plt.figure(figsize=(48, 48))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=100,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(len(PMCAT_List)):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(PMCAT_List[i]), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
    
data = data / 255.
#labels = labels.astype('int')
images = data.reshape(data.shape[0], 28, 28,3)

#plt.figure(figsize=(8,8))
#for i in range(4):
 #   plt.subplot(2,2,i+1)
  #  plt.imshow(images[i])
   # plt.title('truth: {}'.format(labels[i]))


# PCA 
pca = decomposition.PCA(n_components=2)
view = pca.fit_transform(data)
plt.scatter(view[:,0], view[:,1], c=labels, alpha=0.2, cmap='Set1')

#kmeans
kmeans = KMeans(n_clusters=kmeans_cluster, random_state=0).fit(data)
Y = kmeans.labels_ 

#t-SNE 
#view = TSNE(n_components=2, random_state=123).fit_transform(data)
view = TSNE(random_state=123).fit_transform(data)

plt.figure(figsize=(20,10))
plt.scatter(view[:,0], view[:,1], c=labels, alpha=0.5)
plt.xlabel('t-SNE-1')
plt.ylabel('t-SNE-2')

#scatter(view, Y)
scatter_2(view, Label_mcat)
