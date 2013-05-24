
import numpy as np
from skimage.morphology import erosion
from pyVideoDatasets.BackgroundSubtraction import *

try:
	import cv2 as vv
except:
	from pyKinectTools.utils.VideoViewer import VideoViewer
	vv = VideoViewer()

class BasePlayer(object):

	depthIm = None
	colorIm = None
	users = None
	backgroundModel = None
	foregroundMask = None
	prevcolorIm = None

	def __init__(self, base_dir='./', get_depth=True, get_color=False,
				get_skeleton=False, bg_subtraction=False, fill_images=False):

		self.base_dir = base_dir
		self.deviceID = '[]'

		self.get_depth = get_depth
		self.get_color = get_color
		self.get_skeleton =get_skeleton

		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images


	def update_background(self):
		try:
			self.background_model.update(self.depthIm)
			self.mask = self.background_model.get_foreground()
		except:
			self.mask = -1

	def set_background(self, im):
		self.background_model.backgroundModel = im

	def set_bg_model(self, bg_type='box', param=None):
		'''
		Types:
			'box'[param=max_depth]
			'static'[param=background]
			'mean'
			'median'
			'adaptive_mog'
		'''
		if bg_type == 'box':
			self.bg_subtraction = BoxModel(param)
		elif bg_type == 'static':
			if param==None:
				param = self.depthIm
			self.bg_subtraction = StaticModel(depthIm=param)
		elif bg_type == 'mean':
			self.bg_subtraction = MeanModel(depthIm=self.depthIm)
		elif bg_type == 'median':
			self.bg_subtraction = MedianModel(depthIm=self.depthIm)
		elif bg_type == 'adaptive_mog':
			self.bg_subtraction = AdaptiveMixtureOfGaussians(self.depthIm, maxGaussians=5, learningRate=0.01, decayRate=0.001, variance=300**2)
		else:
			print "No background model added"

		self.backgroundModel = self.bg_subtraction.get_model()

	def next(self, frames=1):
		pass

	def get_person(self, edge_thresh=200):
		mask, _, _, _ = extract_people(self.foregroundMask, minPersonPixThresh=5000, gradThresh=edge_thresh)
		self.mask = erosion(mask, np.ones([3,3], np.uint8))
		return self.mask

	def visualize(self, color=True, depth=True, skel=False, text=False):
		# ''' Find people '''
		if skel:
			plotUsers(self.depthIm, self.users)

		if self.get_depth and depth:
			vv.imshow("Depth", (self.depthIm-1000)/float(self.depthIm.max()))
			# vv.imshow("Depth", self.depthIm/6000.)

		if self.get_color and color:
			vv.imshow("Color "+self.deviceID, self.colorIm)
			# vv.putText("Color "+self.deviceID, self.colorIm, "Day "+self.day_dir+" Time "+self.hour_dir+":"+self.minute_dir+" Dev#"+str(self.dev), (10,220))
			# vv.imshow("Color", self.colorIm)

		vv.waitKey(10)

	def run(self):
		pass
