'''
Implements methods for background subtraction
--Adaptive Mixture of Gaussians
--Median Model
--Mean Model
--Static Model
--Box Model
--other utility functions

Colin Lea 2013
'''
import os, time, sys
import numpy as np
import scipy.ndimage as nd
from IPython import embed


def constrain(img, mini=-1, maxi=-1): #500, 4000
	if mini == -1:
		min_ = np.min(img[np.nonzero(img)])
	else:
		min_ = mini
	if maxi == -1:
		max_ = img.max()
	else:
		max_ = maxi

	img = np.clip(img, min_, max_)
	img -= min_
	if max_ == 0:
		max_ = 1
	img = np.array((img * (255.0 / (max_-min_))), dtype=np.uint8)

	img[img==img.max()] = 0

	return img

def extract_people(im, minPersonPixThresh=5000, gradThresh=None):
	'''
	---Paramaters---
	im :
	mask :
	minPersonPixThresh :
	gradThresh :
	gradientFilter :

	---Returns---
	mask :
	userBoundingBoxes :
	userLabels :
	'''
	mask = im!=0
	if gradThresh == None:
		grad_bin = mask
	else:
		gradients = np.gradient(im)
		mag = np.sqrt(gradients[0]**2+gradients[1]**2)
		mask *= mag < gradThresh

	labelIm, maxLabel = nd.label(im*mask)
	connComps = nd.find_objects(labelIm, maxLabel)

	# Only extract if there are sufficient pixels and it is within a valid height/width ratio
	px_count = [nd.sum(labelIm[c]==l) for c,l in zip(connComps,range(1, maxLabel+1))]
	ratios = [(c[1].stop-c[1].start)/float(c[0].stop-c[0].start)for c in connComps]
	du = [float(c[0].stop-c[0].start) for c in connComps]
	dv = [float(c[1].stop-c[1].start) for c in connComps]
	areas = [float(c[0].stop-c[0].start)*float(c[1].stop-c[1].start)for c in connComps]
	# Filter
	usrTmp = [(c,l,px) for c,l,px,ratio,area in zip(connComps,range(1, maxLabel+1), px_count, ratios, areas)
				if ratio < 2 and
					px > minPersonPixThresh and
					px/area > 0.2
			]

	if len(usrTmp) > 0:
		userBoundingBoxes, userLabels, px_count = zip(*usrTmp)
	else:
		userBoundingBoxes = []
		userLabels = []
	userCount = len(userLabels)

	#Relabel foregound mask with multiple labels
	mask = im.astype(np.uint8)*0
	for i,i_new in zip(userLabels, range(1, userCount+1)):
		mask[labelIm==i] = i_new

	return mask, userBoundingBoxes, userLabels, px_count

def get_mean_image(depthImgs):
	mean_ = np.mean(depthImgs, 2)
	mean_ = mean_*(~nd.binary_dilation(mean_==0, iterations=3))

	## Close holes in images
	inds = nd.distance_transform_edt(mean_<500, return_distances=False, return_indices=True)
	i2 = np.nonzero(mean_<500)
	i3 = inds[:, i2[0], i2[1]]
	mean_[i2] = mean_[i3[0], i3[1]] # For all errors, set to avg

	return mean_

def fill_image(im, tol=None):
	## Close holes in images
	inds = nd.distance_transform_edt(im==0, return_distances=False, return_indices=True)
	i2 = np.nonzero(im==0)
	i3 = inds[:, i2[0], i2[1]]
	im[i2] = im[i3[0], i3[1]]

	return im

class BaseBackgroundModel(object):

	backgroundModel = None
	fill_image = False

	def __init__(self, depthIm=None, fill_image=False):

		if depthIm is not None:
			self.backgroundModel = depthIm.copy()
		self.fill_image = fill_image

	def update(self,depthIm):
		self.currentIm = depthIm.copy()

	def get_model(self):
		return self.backgroundModel

	def get_foreground(self, thresh=500):
		max_bg = self.backgroundModel.max()
		return (np.abs(self.backgroundModel - self.currentIm)*(self.currentIm!=0)*(self.currentIm<max_bg)) > thresh


class MedianModel(BaseBackgroundModel):
	'''
	Takes the median of the depths of the past N frames
	'''
	def __init__(self, depthIm, n_images=50, **kwargs):
		super(MedianModel, self).__init__(**kwargs)

		if self.fill_image:
			self.prevDepthIms = fill_image(depthIm.copy())[:,:,None]
		else:
			self.prevDepthIms = depthIm.copy()[:,:,None]
		self.backgroundModel = self.prevDepthIms[:,:,0]
		self.n_images = n_images

	def update(self,depthIm):

		# Add to set of images
		if self.fill_image:
			self.currentIm = fill_image(depthIm.copy())
		else:
			self.currentIm = depthIm.copy()
		self.prevDepthIms = np.dstack([self.prevDepthIms, self.currentIm])

		# Check if too many (or few) images
		imCount = self.prevDepthIms.shape[2]
		if imCount > self.n_images:
			self.prevDepthIms = self.prevDepthIms[:,:,-self.n_images:]

		self.backgroundModel = np.median(self.prevDepthIms, -1)


class MeanModel(BaseBackgroundModel):
	'''
	Averages the depths of the past N frames
	'''
	def __init__(self, depthIm, n_images=50, **kwargs):
		super(MeanModel, self).__init__(**kwargs)

		if self.fill_image:
			self.prevDepthIms = fill_image(depthIm.copy())[:,:,None]
		else:
			self.prevDepthIms = depthIm.copy()[:,:,None]
		self.backgroundModel = self.prevDepthIms[:,:,0]
		self.n_images = n_images

	def update(self,depthIm):

		# Add to set of images
		if self.fill_image:
			self.currentIm = fill_image(depthIm.copy())
		else:
			self.currentIm = depthIm.copy()
		self.prevDepthIms = np.dstack([self.prevDepthIms, self.currentIm])

		# Check if too many (or few) images
		imCount = self.prevDepthIms.shape[2]
		if imCount > self.n_images:
			self.prevDepthIms = self.prevDepthIms[:,:,-self.n_images:]

		self.backgroundModel = np.mean(self.prevDepthIms, -1)

class StaticModel(BaseBackgroundModel):
	'''
	Set an image as a background boundary
	'''
	def __init__(self, **kwargs):
		super(StaticModel, self).__init__(**kwargs)

	def update(self,depthIm):
		self.currentIm = depthIm

class BoxModel(BaseBackgroundModel):
	'''
	Set maximum distance that can be detected as foreground
	'''
	def __init__(self, max_depth=3000, **kwargs):
		super(BoxModel, self).__init__(**kwargs)
		self.max_depth = max_depth

	def update(self,depthIm):
		self.currentIm = depthIm
		self.backgroundModel = depthIm*(depthIm > self.max_depth)

''' Adaptive Mixture of Gaussians '''
class AdaptiveMixtureOfGaussians(BaseBackgroundModel):

	def __init__(self, im, maxGaussians=5, learningRate=0.05, decayRate=0.25, variance=100**2, **kwargs):
		super(AdaptiveMixtureOfGaussians, self).__init__(**kwargs)

		xRez, yRez = im.shape
		self.MaxGaussians = maxGaussians
		self.LearningRate = learningRate
		self.DecayRate = decayRate
		self.VarianceInit = variance
		self.CurrentGaussianCount = 1

		self.Means = np.zeros([xRez,yRez,self.MaxGaussians])
		self.Variances = np.empty([xRez,yRez,self.MaxGaussians])
		self.Weights = np.empty([xRez,yRez,self.MaxGaussians])
		self.Deltas = np.empty([xRez,yRez,self.MaxGaussians])
		self.NumGaussians = np.ones([xRez,yRez], dtype=np.uint8)

		self.Deltas = np.zeros([xRez,yRez,self.MaxGaussians]) + np.inf

		self.Means[:,:,0] = im
		self.Weights[:,:,0] = self.LearningRate
		self.Variances[:,:,0] = self.VarianceInit

		self.Deviations = ((self.Means - im[:,:,np.newaxis])**2 / self.Variances)
		self.backgroundModel = im
		self.currentIm = im

	# @profile
	def update(self, im):

		self.currentIm = im
		self.currentIm[im == 0] = im.max()
		mask = im != 0

		''' Check deviations '''
		self.Deviations = ((self.Means - im[:,:,np.newaxis])**2 / self.Variances) * mask[:,:,None]

		for m in range(self.CurrentGaussianCount):
			self.Deviations[m > self.NumGaussians,m] = np.inf

		Ownership = np.argmin(self.Deviations, -1)
		deviationMin = np.min(self.Deviations, -1)

		createNewMixture = deviationMin > 3
		createNewMixture[np.isinf(deviationMin)] = False
		replaceLowestMixture = np.logical_and(createNewMixture, self.NumGaussians>=self.MaxGaussians)
		createNewMixture = np.logical_and(createNewMixture, self.NumGaussians<self.MaxGaussians)

		''' Create new mixture using existing indices'''
		if np.any(createNewMixture):
			activeset_x, activeset_y = np.nonzero(createNewMixture)
			activeset_z = self.NumGaussians[activeset_x, activeset_y].ravel()

			# print "-------New Mixture------", len(activeset_x)
			self.Means[activeset_x,activeset_y,activeset_z] = im[activeset_x,activeset_y]
			self.Weights[activeset_x,activeset_y,activeset_z] = self.LearningRate
			self.Variances[activeset_x,activeset_y,activeset_z] = self.VarianceInit
			self.NumGaussians[activeset_x,activeset_y] += 1
			Ownership[activeset_x,activeset_y] = activeset_z

		''' Replace lowest weighted mixture '''
		if np.any(replaceLowestMixture):
			activeset_x, activeset_y = np.nonzero(replaceLowestMixture)

			activeset_z = np.argmin(self.Weights[activeset_x, activeset_y,:], -1)

			# print "-------Replace Mixture------", len(activeset_x)
			self.Means[activeset_x,activeset_y,activeset_z] = im[activeset_x,activeset_y]
			self.Weights[activeset_x,activeset_y,activeset_z] = self.LearningRate
			self.Variances[activeset_x,activeset_y,activeset_z] = self.VarianceInit
			Ownership[activeset_x,activeset_y] = activeset_z

		self.CurrentGaussianCount = self.NumGaussians.max()
		# print "Gaussians: ", self.NumGaussians.max()


		''' Update gaussians'''
		for m in range(self.CurrentGaussianCount):
			self.Deltas[:,:,m]		= im - self.Means[:,:,m]
			tmpOwn 					= Ownership==m
			# print "Own:",np.sum(tmpOwn)

			self.Weights[:,:,m]		= self.Weights[:,:,m] 	+ self.LearningRate*(tmpOwn - self.Weights[:,:,m]) - self.LearningRate*self.DecayRate
			tmpWeight 				= tmpOwn*(self.LearningRate/self.Weights[:,:,m])
			tmpMask = (self.Weights[:,:,m]<=0.001)
			tmpWeight[tmpMask] = 0

			self.Means[:,:,m] 		= self.Means[:,:,m] 	+ tmpWeight * self.Deltas[:,:,m]
			# self.Variances[:,:,m] 	= self.Variances[:,:,m] + tmpWeight * (self.Deltas[:,:,m]**2 - self.Variances[:,:,m])

			''' If the weight is zero, reset '''
			if m < np.any(tmpMask):
				if self.CurrentGaussianCount > 1:
					activeset_x, activeset_y = np.nonzero(tmpMask * (self.NumGaussians > m))
					try:
						# self.Variances[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Variances[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						self.Means[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Means[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						self.Weights[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = self.Means[activeset_x, activeset_y, m+1:self.CurrentGaussianCount]
						# self.Variances[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.Means[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.Weights[activeset_x, activeset_y, m:self.CurrentGaussianCount-1] = 0
						self.NumGaussians[activeset_x, activeset_y] -= 1
						# print "Reduce gaussians on slice", m, "max:", self.NumGaussians.max()
					except:
						embed()



		self.backgroundModel = np.max(self.Means, 2)
		# self.backgroundModel = np.nanmax(self.Means, 2)
		# tmp = np.argmax(self.Weights,2).ravel()
		# self.backgroundModel = self.Means[:,:,tmp]



	def get_model(self):
		return self.backgroundModel


