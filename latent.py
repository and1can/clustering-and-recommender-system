import numpy as np
from scipy import io, spatial, misc
import scipy.misc
from matplotlib import pyplot as plt
import random

DEBUG = 0


#print random.__file__

#after write files, then can use this for loop to plot each of the images

	

train_contents = io.loadmat('joke_train.mat')

images = train_contents['train'] #(24983,100)

orig = images

validation = 'validation.txt'

testing = 'query.txt'

class joke(object):
	def __init__(self, d):
		self.d = d
		self.u = [] 			#(d, 100)
		self.v = []				#(24983, d)

	def train(self, data):
		image = np.nan_to_num(data)
		mean = np.mean(image, axis=0)
		image = image - mean
		U, s, V = np.linalg.svd(image, full_matrices=False)
		sort = np.argsort(s)[::-1]						#[::-1] <-- to reverse the order
		#print "image", image, "mean", mean, "new image", image
		#print U.shape, V.shape, s.shape
		#print "sort", sort, "np.argsort", np.argsort(s)
		d = self.d
		U = U[:, sort]
		V = V.T[:, sort]
		S = np.diag(s[sort])
		self.u = np.dot(U[:,:d], np.sqrt(S[:d,:d]))
		self.v = np.dot(np.sqrt(S[:d,:d]), V[:,:d].T)
		#print "v", self.v.shape, "u", self.u.shape
		print "MSE for " + str(self.d)
		mse = np.nansum((np.dot(self.u, self.v) - image)**2)
		print mse

	def predict(self, file):
		pred = []
		f = open(file, 'r')
		for line in f:
			line = line.split(',')
			user = int(line[0]) - 1
			joke_num = int(line[1]) - 1 # - 1 because of how python starts with 0 for index
			joke_avg_rating = np.dot(self.u[user,:], self.v[:,joke_num]) 
		
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


j = joke(2)
j.train(images)
prediction = j.predict(validation)
print "accuracy for meansq " + str(2), accuracy(prediction, validation)

j = joke(5)
j.train(images)
prediction = j.predict(validation)
print "accuracy for meansq " + str(5), accuracy(prediction, validation)

j = joke(10)
j.train(images)
prediction = j.predict(validation)
print "accuracy for meansq " + str(10), accuracy(prediction, validation)

j = joke(20)
j.train(images)
prediction = j.predict(validation)
print "accuracy for meansq" + str(20), accuracy(prediction, validation)

class adv(object):
	def __init__(self, d, threshold, grad, lamb):
		self.d = d
		self.u = [] 			#(d, 100)
		self.v = []				#(24983, d)
		self.threshold = threshold
		self.grad = grad
		self.lamb = lamb
	def train(self, data):
		image = np.nan_to_num(data)
		mean = np.mean(image, axis=0)
		image = image - mean
		U, s, V = np.linalg.svd(image, full_matrices=False)
		sort = np.argsort(s)[::-1]						#[::-1] <-- to reverse the order
		# print "image", image, "mean", mean, "new image", image
		# print U.shape, V.shape, s.shape
		# print "sort", sort, "np.argsort", np.argsort(s)
		d = self.d
		U = U[:, sort]
		V = V.T[:, sort]
		S = np.diag(s[sort])
		self.u = np.dot(U[:,:d], np.sqrt(S[:d,:d]))
		self.v = np.dot(np.sqrt(S[:d,:d]), V[:,:d].T)
		# print "v", self.v.shape, "u", self.u.shape
		# print "loss for " + str(self.d)
		#print "dot of u and v ", np.dot(self.u, self.v).shape
		print (np.dot(self.u, self.v) - orig)**2
		print np.sum(np.multiply(self.u, self.u))
		print np.sum(np.multiply(self.v, self.v))
		loss = np.nansum((np.dot(self.u, self.v) - orig)**2) + self.lamb* (np.sum(np.multiply(self.u, self.u)) + np.sum(np.multiply(self.v, self.v)))
		print "first loss", loss
		iteration = 0
		prev = loss
		while True:
			#print "u", self.u, "u shape", self.u.shape
			#print "v", self.v, "v shape", self.v.shape
			#print "dot product", np.dot(self.u, self.v), "dot shape", np.dot(self.u, self.v).shape
			#print "image", image, "image shape", image.shape
			#print "subtract image from dot product", np.dot(self.u, self.v) - image
			#self.u -= 2*(np.dot(np.nan_to_num((np.dot(self.u, self.v) - orig)), self.v.T) + self.lambd*self.u)
			if (iteration ==501):
				print "converged on iteration ", iteration, " with loss of ", loss
				return

			self.u -= 2 * self.grad*(np.dot((np.nan_to_num(np.dot(self.u, self.v) - orig)), self.v.T) + self.lamb * self.u)
			#self.v -= 2*(np.dot(self.u.T, np.nan_to_num(np.dot(self.u, self.v) - orig)) + self.lambd*self.v)
			self.v -= 2 * self.grad*(np.dot(self.u.T, np.nan_to_num((np.dot(self.u, self.v) - orig))) + self.lamb * self.v)
			#print "new u", self.u
			#print "new v", self.v
			loss = np.nansum((np.dot(self.u, self.v) - orig)**2) + self.lamb* (np.sum(np.multiply(self.u, self.u)) + np.sum(np.multiply(self.v, self.v)))
			if (np.abs(prev - loss) < self.threshold):
				print "converged on iteration ", iteration, " with loss of ", loss
				return 
			if ((iteration % 100) == 0):
				print "iteration ", iteration
				print "loss of ", loss
				print "prev - loss", np.abs(prev - loss)
				prediction = j.predict(validation)
				print "accuracy for", iteration, "is", accuracy(prediction, validation)

			prev = loss
			iteration += 1
		

	def predict(self, file):
		pred = []
		f = open(file, 'r')
		for line in f:
			line = line.split(',')
			user = int(line[0]) - 1
			joke_num = int(line[1]) - 1 # - 1 because of how python starts with 0 for index
			joke_avg_rating = np.dot(self.u[user,:], self.v[:,joke_num]) 
		
			if joke_avg_rating > 0:
				pred.append(1)
			else:
				pred.append(0)
		return pred

	def test(self, file):
		pred = []
		f = open(file, 'r')
		w = open('kaggle_submission.txt', 'w')
		w.write('Id,Category\n')
		for line in f:
			line = line.split(',')
			user = int(line[1]) - 1
			joke_num = int(line[2]) - 1 # - 1 because of how python starts with 0 for index
			joke_avg_rating = np.dot(self.u[user,:], self.v[:,joke_num]) 
		
			if joke_avg_rating > 0:
				w.write(str(line[0]) + ',' + str(1) + '\n')
			else:
				w.write(str(line[0]) + ',' + str(0) + '\n')
		w.close()
		return pred


j = adv(10, 0.1, 0.0001, 100)
j.train(images)
prediction = j.predict(validation)
print "accuracy for sparseAdjusted ", accuracy(prediction, validation)
j.test(testing)











