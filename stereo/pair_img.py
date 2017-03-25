#!/usr/bin/python

import numpy as np
from PIL import Image
import pdb
#import ox
import cv2

def match(desc_1, desc_2, thresh, patch_wid):
	#for each point of interest in img 1 evaluate a match score for each point in img 2 through NORMALIZED CROSS-CORRELATION

	n=patch_wid**2.0
	match=[]
	mean_ncc=[]
	for a in range(len(desc_1)):
		desc_gr_1=desc_1[a]
		match_b=[]
		ncc_m=[]
		for b in range(len(desc_2)):
			desc_gr_2=desc_2[b]

			ncc_mean=0
			ncc=-np.ones((len(desc_gr_1), len(desc_gr_2)))
			for i in range(len(desc_gr_1)):
				for j in range(len(desc_gr_2)):
					I_1=desc_gr_1[i]
					I_2=desc_gr_2[j]
					if I_1.shape[0]!=10 or I_2.shape[0]!=10 or I_2.shape[1]!=10 or I_1.shape[1]!=10: continue
					std_1=np.std(I_1)
					std_2= np.std(I_2)
					if std_2==0.0: std_2=0.01
					if std_1==0.0: std_1=0.01
					d1=(I_1-np.mean(I_1))/std_1
					d2=(I_2-np.mean(I_2))/std_2
					ncc_val=np.sum(np.multiply(d1,d2))/(n-1)
					if ncc_val> thresh: 
						ncc[i,j]=ncc_val
						ncc_mean+=ncc_val

			ncc_mean=ncc_mean/np.sum(ncc.shape)

			ncc_m.append(ncc_mean)
			match_b.append(ncc)
			#ndx=np.argsort(-ncc)
			#pdb.set_trace()
			#matchscores=ndx[:,0]

			#match_b.append(matchscores)
		match.append(match_b)
		mean_ncc.append(ncc_m)
		#pdb.set_trace()
	return match, mean_ncc

def best_n(ncc, n):
#find best n- matches

	pos=np.zeros((n,2))
	a=0
	while a<n:
		ncc_max=0
		for i, row in enumerate(ncc):
			for j, el in enumerate(row):
				if el>ncc_max:
					ncc_max=el
					pos_max=np.array([i,j])
				#else: pos_max=np.array([-1,-1])

		pos[a,0]=pos_max[0]
		pos[a,1]=pos_max[1]
		ncc[pos_max[0]][pos_max[1]]=0
		a+=1
	return pos

def appendimages(im1, im2):
	#returns new im that stack two images horizontally
	
	return np.concatenate((im1,im2),axis=1)

def get_data(matches, pos, kp_groups_im):

	data=[]
	for k,j in enumerate(pos[:, 0]):
		i= pos[k,1]
		#print i,j
		ncc=matches[int(j)][int(i)]
		data_ij=[]
		for row in range(len(ncc[:,0])):
			idx=np.argsort(ncc[row,:])
			if idx[1]>1.0: 
			#for el_ix, el in enumerate(ncc[row, :]):
				#if el !=1.0:
					kp_1=kp_groups_im[0][int(j)][row]
					kp_2=kp_groups_im[1][int(i)][idx[1]]
					data_ij.append([kp_1, kp_2])
		data.append(data_ij)
	return data







