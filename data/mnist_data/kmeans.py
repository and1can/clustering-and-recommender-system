import numpy as np
from scipy import io, spatial, misc
import scipy.misc
from matplotlib import pyplot as plt
import random






	

train_contents = io.loadmat('images.mat')

images = train_contents['images']


pixel_images = images[0,:,:]
for i in range(1, 28):

    pixel_images = np.vstack([pixel_images, images[i,:,:]])



pixel_images = pixel_images.T


class KMeans(object):
    def __init__(self, k=10, iters=300):
        self.k = k
        self.centroids = {}
        self.iters = iters
        
        
        
            
    def train(self, data):
        #initialize the centroids randomly with 784 pixels with values up to 255
        for i in range(self.k):
            self.centroids[i] = np.random.choice(256, 784) #256 because the max pixel in 255

        iters = 0
        
        while iters < self.iters:
            clusters = {}
            J = 0.0
            index = 0
            #if ((iters % 10) == 0): 
            print "training", iters
            for s in data:
                minDist = 3000000    #initialize to number bigger than 255*784 = 200,175
                label = -1
                # find the label c that has smalled euclidean distance with data point
                for c in self.centroids:
                    dist = spatial.distance.euclidean(s, self.centroids[c])
                    if dist < minDist: 
                        minDist = dist
                        label = c
                        
                #create new group if smallest label is not in clusters
                if label not in clusters:
                    clusters[label] = []
                
                #add the data to cluster it belongs to
                clusters[label].append(index)
                
                index += 1
            # take mean of each cluster to find the new centroid
            # increment the loss to figure out new loss
            for c in clusters:
                self.centroids[c] = np.mean(data[clusters[c]], axis=0)
                for g in data[clusters[c]]:
                    J += spatial.distance.euclidean(g, self.centroids[c])
                    
            print "Loss is", J, "for iteration ", iters
                
            iters += 1

        print "final loss of", J, "for iteration ", iters

iters = 50

km5 = KMeans(5, iters)
km5.train(pixel_images)

for n in km5.centroids:
    np.savetxt('q1/km5/' + str(n), km5.centroids[n].reshape(28,28))

#after write files, then can use this for loop to plot each of the images
for i in range(5):
    p = np.loadtxt('q1/km5/' + str(i))
    
    plt.imshow(p)
    plt.savefig('q1/km5/' + str(i))


km10 = KMeans(10, iters)
km10.train(pixel_images)

for n in km10.centroids:
    np.savetxt('q1/km10/' + str(n), km10.centroids[n].reshape(28,28))

#after write files, then can use this for loop to plot each of the images
for i in range(10):
    p = np.loadtxt('q1/km10/' + str(i))
    
    plt.imshow(p)
    plt.savefig('q1/km10/' + str(i))


km20 = KMeans(20, iters)
km20.train(pixel_images)

for n in km20.centroids:
    np.savetxt('q1/km20/' + str(n), km20.centroids[n].reshape(28,28))

#after write files, then can use this for loop to plot each of the images
for i in range(20):
    p = np.loadtxt('q1/km20/' + str(i))
    
    plt.imshow(p)
    plt.savefig('q1/km20/' + str(i))




