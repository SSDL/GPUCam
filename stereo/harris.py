#!/usr/bin/python

import numpy as np
from PIL import Image
import pdb
#import ox
import cv2
from scipy.ndimage.filters import *
from matplotlib import figure
from pylab import *

def harris_resp(im, sig_x, sig_y):
	im_x=np.zeros(im.shape)
	im_y=np.zeros(im.shape)
	gaussian_filter(im, (sig_x,sig_y), (0,1), im_x)
	gaussian_filter(im, (sig_x,sig_y), (1,0), im_y)

	Wxx=gaussian_filter(im_x*im_x, sig_x)
	Wxy=gaussian_filter(im_x*im_y, (sig_x+sig_y)/2.0)
	Wyy=gaussian_filter(im_y*im_y, sig_y)

	return (Wxx*Wyy-Wxy**2.0)/(Wxx+Wyy)

def get_harris_pts(harris_im, min_d, thresh):
	corner_thresh=harris_im.max()*thresh
	harris_im_t=(harris_im > corner_thresh)
	coord = np.array(harris_im_t.nonzero()).T
	candidate_val=[harris_im[c[0],c[1]] for c in coord]
	index=np.argsort(candidate_val)
	
	allow_loc=np.zeros(harris_im.shape)
	allow_loc[min_d:-min_d, min_d:-min_d]=1

	filtered_coord=[]
	for i in index:
		if allow_loc[coord[i,0], coord[i,1]]==1:
			filtered_coord.append(coord[i])
			allow_loc[(coord[i,0]-min_d):(coord[i,0]+min_d), (coord[i,1]-min_d):(coord[i,1]+min_d)]=0
	return filtered_coord

def plot_harris_pt(im, filtered_coord):

	figure()
	gray()
	imshow(im)
	plot([p[1] for p in filtered_coord],[p[0] for p in filtered_coord],'*')
	axis('off')
	show()


