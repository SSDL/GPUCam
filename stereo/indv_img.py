#!/usr/bin/python

import sift
import numpy as np
from PIL import Image
import pdb
import cv2
import scipy
from matplotlib import pyplot as plt
import random

sift=cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02, edgeThreshold=15)

def get_kp(im_name, border):
	img=cv2.imread(im_name)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	n,m=gray.shape

	denoised = cv2.fastNlMeansDenoising(gray,None,15,7,21)
	#plt.imshow(img2)
	#plt.show()

	kp=sift.detect(denoised,None) #des is descriptor
	#print len(kp)
	a= np.size(kp)
	i=0	
	while i >a:
		if kp[i].pt[0]>border and kp[i].pt[0]<(m-border) and kp[i].pt[1]>border and kp[i].pt[0]<(n-border):
			i+=1
		else:
			kp=kp[:i]+kp[i+1:]
			a-=1

	print len(kp)
		
	
	#img=cv2.drawKeypoints(gray,kp, gray)
	#cv2.imwrite('bla.jpg',gray)
	#pdb.set_trace()

	kp_loc_x=np.zeros(np.size(kp))
	kp_loc_y=np.zeros(np.size(kp))
	for i in range(np.size(kp)):
		kp_loc_x[i]=kp[i].pt[0]
		kp_loc_y[i]=kp[i].pt[1]
	kp_loc=np.c_[kp_loc_x, kp_loc_y]
	
	return gray, kp, kp_loc

def kp_dis_filter(d_thresh, kp, kp_loc, pt_i):

	x0, y0=kp_loc[pt_i]
	j=0
	a=len(kp_loc[:,0])
	while j < a:
		if j != pt_i:
			x,y=kp_loc[j]
			r=(x-x0)**2.0+(y-y0)**2.0
			if r>=d_thresh:
				kp_loc=np.delete(kp_loc, j, 0)
				kp=kp[:j]+kp[j+1:]
				a-=1
			else: j+=1
	return kp_loc, kp

def kp_circle(r_thresh, kp, kp_loc, i):
	(x0, y0)=kp_loc[i]
	idx_group=np.array([i])
	for j in range(len(kp)):
		if j!=i and kp[j]!=0 and kp[i]!=0:
			x,y= kp_loc[j]
			r=np.sqrt((x-x0)**2.0+(y-y0)**2.0)
			#kpprint r
			if r<r_thresh:
				idx_group=np.hstack([idx_group, j])
	if len(idx_group)>5:
		return idx_group
	else:return 0

def print_kp(im, kp_groups, im_name):#im=gray
	color=[(300,0,0), (0,150,0), (0,0,200), (150, 150, 0), (0, 150, 150), (150, 0, 150), (100, 100, 100)]
	if len(kp_groups)> len(color):
		print 'amount of distinct groups is big (>7) - be careful'
		diff=len(kp_groups)-len(color)
		for i in range(diff):
			color.append((np.random.randint(0,150), np.random.randint(0,150), np.random.randint(0,150)))
	for i in range(len(kp_groups)):
		im=cv2.drawKeypoints(im,kp_groups[i],im, color[i])
		#pdb.set_trace()
		cv2.imwrite(im_name[:-4]+'_kp.jpg',im)
	#print im.shape
	return im

def desc_image(im, kp_groups, patch_wid):
	desc=[]
	for i in range(len(kp_groups)):
		kp_group=kp_groups[i]
		group_desc=[]
		for kp in kp_group:
			a,b,c,d=[kp.pt[0]-patch_wid,kp.pt[0]+patch_wid, kp.pt[1]-patch_wid,kp.pt[1]+patch_wid]#.flatten
			#print desc_kp
			#print a, b, c, d
			desc_kp=im[int(c):int(d), int(a):int(b)]
			#print desc_kp
			group_desc.append(desc_kp)
		desc.append(group_desc)
	return desc
	
def desc_im2(im, kp_groups, patch_wid):
	desc=[]
	for i in range(len(kp_groups)):
		kp_group=kp_groups[i]

		if len(kp_group)>40: kp_group=random.sample(kp_group, int((.5)*len(kp_group)))
		group_desc=[]
		for kp in kp_group:
			a,b,c,d=[kp[0][0]-patch_wid,kp[0][0]+patch_wid, kp[0][1]-patch_wid,kp[0][1]+patch_wid]#.flatten
			#print desc_kp
			#print a, b, c, d
			desc_kp=im[int(c):int(d), int(a):int(b)]
			#print desc_kp
			group_desc.append(desc_kp)
		desc.append(group_desc)
	return desc
	
