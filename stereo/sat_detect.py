#!/usr/bin/python

import numpy as np
from PIL import Image
import pdb
#import ox
import cv2
from fit_line import *
from pylab import *

#here I is the image/image section, kp is the kp to consider cneter of a (2n+1)^2 window with thresh the threshhold for the cornerness fct 
def write_dS(I, k):

	#wnat a gaussian smoothign bef?
	#dgdx=1./(2*np.pi*sig**2.0)* np.exp(-1./(2*sig)*(x[0]**2.+x[1]**2.))*(-x[0]/sig)
	#dgdy=1./(2*np.pi*sig**2.0)* np.exp(-1./(2*sig)*(x[0]**2.+x[1]**2.))*(-x[1]/sig)
	#cv2.gaussianblur()

	
	Sx=cv2.Sobel(I, -1, 1, 0, 3)
	Sy=cv2.Sobel(I, -1, 0, 1, 3)
	dS=np.zeros(Sx.shape)
	for i in range(Sx.shape[0]):
		for j in range(Sx.shape[1]):
			dS[i,j]=np.sqrt(Sx[i,j]**2.0+Sy[i,j]**2.)
			if dS[i,j]< 100: dS[i,j]=0
	cv2.imwrite('im_'+k+'.jpg',dS)
	return dS

#in: list of kps-groups
#out: lits of the midpoint coordinates for each group [list of list of int]

def find_midpt_kp(kps):
	mid=[]
	for gr in kps:
		mid_x=0
		mid_y=0
		for kp in gr:
			x=kp.pt[0]
			y=kp.pt[1]
			mid_x+=x
			mid_y+=y
			mid_gr=[int(mid_x/len(gr)), int(mid_y/len(gr))]
		mid.append(mid_gr)
	return mid

def contour2(gray, kps, k, thresh_distance):
	
	denoised= cv2.fastNlMeansDenoising(gray,None,15,7,21)

	#out: contour[0] - new image
	#contour[1]- pts in contour lines
	#contour[2]- hierarchy
	ret, thresh = cv2.threshold(gray, 127, 255, 0)
	contour=cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
	test=cv2.drawContours(gray, contour[1], -1, (255,0, 0), 3)

	mid=find_midpt_kp(kps)


	#filter for length and distance form object : store endpoints or entire line(##) (easier axess):
	lines=[]
	for line in contour[1]:
		if len(line)>3:
			new_line=[]#lines[[line[0][0,0] ,line[0][0,1] ], [line[-1][0,0], line[-1][0,1]] ]
			#make executable if you wnat to store the entire line
			for pt in line:
				el=np.array([pt[0,0],pt[0,1]])
				new_line.append(el)
			radii=[min([np.linalg.norm(np.sum([-np.array(new_line[0]), np.array(midpt)], axis=0)) for el in new_line ]) for midpt in mid]
			if any(radius < thresh_distance for radius in radii): lines.append(new_line)

	#write
	rl=['R','L']
	cv2.imwrite('contour_'+rl[int(k)]+'.jpg',test)
	return lines, contour[1]


def linefitting(lines):

	params = {'MIN_SEG_LENGTH': 0.00, 'LINE_POINT_DIST': 3.0,
            'MIN_POINTS_PER_SEG': 10, 'MAX_P2P_DIST':400}
	#spliting
	i=0
	len_lin=len(lines)
	while i < len_lin:
		line=lines[i]
		theta=convert_to_polar(line)[:,0]
		rho=convert_to_polar(line)[:,1]
		alpha, r= FitLine(theta, rho)
		idx=FindSplit(theta, rho, alpha, r, params)
		if idx!=-1:
			lines[i]=line[:idx]
			lines.insert(i+1, line[idx:])
			len_lin+=1
		else: i+=1
	return lines

def contour(I, kps, k, thresh_radius=100, intensity=70):

	non_zero=[]
	rl=['R','L']
	#I= cv2.fastNlMeansDenoisingColored(I,None,10,10,7,21)

	Sx=cv2.Sobel(I, -1, 1, 0, 3)
	Sy=cv2.Sobel(I, -1, 0, 1, 3)
	dS=np.zeros(Sx.shape)
	for i in range(Sx.shape[0]):
		for j in range(Sx.shape[1]):
			dS[i,j]=np.sqrt(Sx[i,j]**2.0+Sy[i,j]**2.)
			if dS[i,j]< intensity: dS[i,j]=0
	cv2.imwrite('contour_'+rl[int(k)]+'.jpg',dS)

#	Sx=cv2.Sobel(dS, -1, 1, 0, 3)
#	Sy=cv2.Sobel(dS, -1, 0, 1, 3)
#	dS=np.zeros(Sx.shape)
#	for i in range(Sx.shape[0]):
#		for j in range(Sx.shape[1]):
#			if Sx[i, j]>0. and Sy[i, j]>0.:
#				dS[i,j]=np.sqrt(Sx[i,j]**2.0+Sy[i,j]**2.)
#				if dS[i,j]< 100: dS[i,j]=0
#	cv2.imwrite('im_'+k+'.jpg',dS)

	#find midpt fo each gr:
	mid=[]
	for gr in kps:
		mid_x=0
		mid_y=0
		for kp in gr:
			x=kp.pt[0]
			y=kp.pt[1]
			mid_x+=x
			mid_y+=y
			mid_gr=[int(mid_x/len(gr)), int(mid_y/len(gr))]
		mid.append(mid_gr)

	#set all points equal:
	for i in range(dS.shape[1]):
		for j in range(dS.shape[0]):
			if dS[j, i] > intensity:
				radii=[np.sqrt((mid_pt[0]-i)**2.+(mid_pt[1]-j)**2.) for mid_pt in mid]
				if any(radius < thresh_radius for radius in radii):
					dS[j,i]=100
					non_zero.append(np.array([i,j]))
				else: dS[j, i]=0
	cv2.imwrite('contour_'+rl[int(k)]+'.jpg',dS)
	contour=cv2.findContours(dS, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

	polar=convert_to_polar(non_zero)
	lines=[polar]

 	params = {'MIN_SEG_LENGTH': 0.00, 'LINE_POINT_DIST': 1.0,
            'MIN_POINTS_PER_SEG': 10, 'MAX_P2P_DIST':400}

	#spliting
	i=0
	len_lin=len(lines)
	while i < len_lin:
		line=lines[i]
		theta=np.array([pt[0] for pt in line])
		rho=np.array([pt[1] for pt in line])
		alpha, r= FitLine(theta, rho)
		pdb.set_trace()
		idx=FindSplit(theta, rho, alpha, r, params)
		if idx!=-1:
			lines[i]=line[:idx]
			lines.insert(i+1, line[idx:])
			len_lin+=1
		else: i+=1

	pdb.set_trace()
	#check out next line
	i=0
	lin_len=len(lines)-1
	while i < lin_len:
		line=lines[i]
		next_line=lines[i+1]
		start_idx=int(next_line[0][2])
		end_idx=int(line[-1][2])
		dist=np.linalg.norm(np.sum([non_zero[end_idx],-1.*non_zero[start_idx]], axis=0))
		if dist < 2.5:
			for el in next_line:
				lines[i].append(el) #merge
			lines=np.delete(lines, i+1)
			lin_len-=1
		else: i+=1
	print len(lines)

	#mix and macth: does anz line macth?
	i=0
	lin_len=len(lines)-1
	while i < lin_len:
		line=lines[i]
		start_idx=int(line[0][2])
		end_idx=int(line[-1][2])
		for j, next_line in enumerate(lines):
			if j==i: continue
			start_next=int(next_line[0][2])
			end_next=int(next_line[-1][2])

			dist1=np.linalg.norm(np.sum([non_zero[end_idx],-1.*non_zero[start_next]], axis=0))
			print dist1
			dist2=np.linalg.norm(np.sum([non_zero[end_next],-1.*non_zero[start_idx]], axis=0))
			print dist2
			
			if dist1< 1.5:
				for el in lines[j]:
					lines[i].append(el) #merge
				lines=np.delete(lines, j)
				lin_len-=1
			elif dist2< 1.5:
				for el in lines[i]:
					lines[j].append(el) #merge
				lines[i]=lines[j]
				lines=np.delete(lines, j)
				lin_len-=1
			else: i+=1

	#delete artifacts
	del_el=[]
	for i, line in enumerate(lines):
		#print line
		n=len(line)
		if n< 20: del_el.append(i)
	for i in reversed(del_el):
		lines=np.delete(lines, i)

	#check plot
	for line in lines:
		for pol in line:
			i=int(pol[2])
			pt=non_zero[i]
			dS[pt[1], pt[0]]=300
	cv2.imwrite('im_'+k+'.jpg',dS)

	return lines, dS, non_zero


#	pdb.set_trace()
#	for start_kp in kps:
#		flag=True
#		start_pt=np.array([start_kp.pt[0], start_kp.pt[1]])
#		#mid_pt=np.array([mid_kp.pt[0], mid_kp.pt[1]])
#		
#		contour_pts=[start_pt]
#		n_pt=0
#		while flag and n_pt<800:
#			#draw square:
#			square=[]
#			for j in range(n):
#				if dS[start_pt[1]+j, start_pt[0]+n]==100: square.append(np.array([start_pt[1]+j, start_pt[0]+n, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]-j, start_pt[0]+n]==100: square.append(np.array([start_pt[1]-j, start_pt[0]+n, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]+j, start_pt[0]-n]==100: square.append(np.array([start_pt[1]+j, start_pt[0]-n, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]-j, start_pt[0]-n]==100: square.append(np.array([start_pt[1]-j, start_pt[0]-n, np.sqrt(j**2.+n**2.)]))
#	
#				if dS[start_pt[1]+n, start_pt[0]+j]==100: square.append(np.array([start_pt[1]+n, start_pt[0]+j, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]+n, start_pt[0]-j]==100: square.append(np.array([start_pt[1]+n, start_pt[0]-j, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]-n, start_pt[0]+j]==100: square.append(np.array([start_pt[1]-n, start_pt[0]+j, np.sqrt(j**2.+n**2.)]))
#				if dS[start_pt[1]-n, start_pt[0]-j]==100: square.append(np.array([start_pt[1]-n, start_pt[0]-j, np.sqrt(j**2.+n**2.)]))
#			d_min=[100, 0]
#			for i in range(len(square)):
#				if square[i][2]<d_min[0]: d_min=[square[i][2], i]
#			if d_min[0]!=100: 
#				start_pt=np.array([square[d_min[1]][1], square[d_min[1]][0]])
#				contour_pts.append(start_pt)
#				n_pt+=1
#			else: flag=False
#			print flag

#		if len(contour_pts)>5: contour.append(contour_pts)
	#	for j in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
	#		if dS[start_pt[1]+j, start_pt[0]+n]==100:
#
#
#		for i in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
#			for j in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
#				if dS[start_pt[1]+j, start_pt[0]+i]==100:
#					start_pt=np.array([start_pt[0]+i, start_pt[1]+j])
#					contour_pts.append(start_pt)
#		if len(contour_pts)>5: contour.append(contour_pts)
	
#	pdb.set_trace()
#	sat=dS#np.zeros(dS.shape)
#	for i in range(len(contour)):
#		for pt in contour[i]:
#			sat[pt[1], pt[0]]= 300.0
#	cv2.imwrite('contour.jpg', sat)







