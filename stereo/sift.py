#!/usr/bin/python

import numpy as np
import os
import cv2
#import vlfeat
from PIL import Image

#sift=cv2.SIFT()

def process_im(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
	if imagename[-3:]!='pgm':
		im=Image.open(imagename).convert('L')
		im.save('70.pgm')
		imagename='70.pgm'
	
	cmmd=str("sift "+imagename+" --output=" +resultname+""+params)
	os.system(cmmd)

def read_features_from(filename):
	f=np.loadtxt(filename)
	return f[:,:4], f[:,4:]

def write_features(filename,locs,desc):
	np.savetxt(filename, np.hsatck((locs,desc)))

def plot_features(im, locs, circle=False):
	def draw_circle(c,r):
		t=arange(0,1.01,.01)*2*np.pi
		x=r*np.cos(t)+c[0]
		y=r*np.sin(t)+c[1]
		plot(x,y,'b',linewidth=2)
	imshow(im)
	if circle:
		for p in locs:
			draw_circle(p[:2],p[2])
	else:
		plot(locs[:,0],locs[:,1], 'ob')
	axis('off')

