import numpy as np
from scipy import io, spatial, misc
import scipy.misc
from matplotlib import pyplot as plt
from scipy import io, spatial, misc
import random


train_contents = io.loadmat('joke_train.mat')

train = train_contents['train']

validation = 'validation.txt'

class recommend_avg(object):
	def __init__ (self, data):
		self.avg = np.array([np.nanmean(data, axis=0)]).T
		#print self.avg.shape

	def predict(self, file):
		pred = []
		f = open(file, 'r')
		for line in f:
			line = line.split(',')
			joke_num = int(line[1]) - 1 # - 1 because of how python starts with 0 for index
			joke_avg_rating = self.avg[joke_num][0]
		
			if joke_avg_rating > 0:
				pred.append(1)
			else:
				pred.append(0)
		return pred


def accuracy(prediction, validation):
	correct = 0.0
	total = 0
	f = open(validation, 'r')
	for line in f:
		line = line.split(',')
		if (int(line[2])) == int(prediction[total]):
			correct += 1
		total += 1
	return correct/float(total)

ra = recommend_avg(train)
prediction = ra.predict(validation)
print "standard accuracy is", accuracy(prediction, validation)


class advanced(object):
	def __init__(self, data, k):
		#self.data = np.array([np.nan_to_num(data)])
		self.data = np.nan_to_num(data)  #(24983, 100)
		self.k = k
		self.avg = {}
		#print "data", self.data, self.data[0][0]			#looks good because nan replaced by 0's

	def train(self):
		#print "time to train", "data shape", self.data.shape, self.data
		avg = self.avg
		#distLst = []
		#print "avg init", avg
		avgcluster = []
		#print "data shape", self.data.shape
		for i in range(101):								#101 because validation is 100
		#for i in range(self.data.shape[0]):                 #need to start at 0, else first 1 is not counted
			for j in range(self.data.shape[0]):
				if (i != j):
					#print "calc euclidean for", i, j
					#for uniitialized case, store into dictionary the user and also k,2 values that are bigger than the 
					#biggest euclidean distance possible
					#having the biggest euclidean distance possible will allow all initialized values to be replaced
					if i not in avg:
						#print "key" + str(i) + " not in dictionary"
						add = np.empty([self.k, 2])
						#print "initialize this", add
						#print "dim of initialize", add.shape
						add.fill(201)  #since the largest distance is sqrt((20^2)*100) is 200
						#print "change all in add to particular value", add
						#print "add key", i
						avg.update({i:add})
					#check euclidean distance and self.avg when have a better euclidean distance
					else:
						# print "i", self.data[i].shape
						# print "i is", self.data[i]
						# print "j is", self.data[j]
						#dist = np.linalg.norm((self.data[i], self.data[j]))
						dist = spatial.distance.euclidean(self.data[i], self.data[j])
						#distLst.append(dist)
						usrData = avg.get(i)
						usrDist = usrData[:,0]
						evict = np.amax(usrDist)
						index = np.argmax(usrDist)
						if evict > dist:
							#usrData[index] = np.asarray([dist, j])
							avg[i][index] = [dist, j] 
				
			
				#if (j== 101):
			#		break
			#if (i==101):
		    #		break
			#print avg[i], "for ", i
			close = avg.get(i)[:,1].astype(int)
			#print self.data[close]
			#print self.data[close].shape, "shape of closest neighbors"
			cluster = np.mean(self.data[close],axis=0)
			#print cluster.shape, "after mean"
			avgcluster.append(cluster)
			print "on", i
		self.avg = avgcluster
		#print len(self.avg), "shape of avg cluster"


	def predict(self, file):
		pred = []
		f = open(file, 'r')
		for line in f:
			line = line.split(',')
			joke_num = int(line[1]) - 1 # - 1 because of how python starts with 0 for index
			user = int(line[0]) - 1
			#print "self.avg[user]", self.avg[user]
			#print "joke num", joke_num
			#print "value is", self.avg[user][joke_num]
			#print "user", user, "joke num", joke_num
			joke_avg_rating = self.avg[user][joke_num]
		
			if joke_avg_rating > 0:
				pred.append(1)
			else:
				pred.append(0)
		return pred

			#print "max dist", max(distLst), "min dist", min(distLst)

adv = advanced(train, 10)
#print "adv object", adv.data, adv.k, adv.avg
adv.train()
prediction = adv.predict(validation)
print "accuracy advanced 10", accuracy(prediction, validation)

adv = advanced(train, 100)
#print "adv object", adv.data, adv.k, adv.avg
adv.train()
prediction = adv.predict(validation)
print "accuracy advanced 100", accuracy(prediction, validation)

adv = advanced(train, 1000)
#print "adv object", adv.data, adv.k, adv.avg
adv.train()
prediction = adv.predict(validation)
print "accuracy advanced 1000", accuracy(prediction, validation)
