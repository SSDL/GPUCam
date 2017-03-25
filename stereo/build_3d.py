#!/usr/bin/python

import numpy as np
from PIL import Image
import pdb
import cv2

from mpl_toolkits.mplot3d import axes3d


#-----------------------------------------------
def find_midpt(line):
	mid=np.array([0.,0.])
	for pt in line:
		mid[0]+=pt[0][0]
		mid[1]+=pt[0][1]
	mid=(1./len(line))*mid
	return mid

#------------------------------------------------
#filters for start and endpoint of a line IF straight line/ points that are farest away from each other
def find_keypoints(line):
	mid=find_midpt(line)
	dis=[np.sqrt((pt[0][0]-mid[0])**2.+(pt[0][1]-mid[1])**2.) for pt in line]
	start=dis.index(max(dis))
	mid=line[start][0]
	dis=[np.sqrt((pt[0][0]-mid[0])**2.+(pt[0][1]-mid[1])**2.) for pt in line]
	end=dis.index(max(dis))
	return [line[start], line[end]]

#-----------------------------------------------
#estimate distance approx/ roughly
#input:
#output: z (distance) in m

def first_dist_est(pos, lines, b=90*10**(-3.), f=16*10**(-3.), ps=5.3*10**(-6.), offset=300):
	u_r=find_midpt(lines[0][int(pos[0][0])])[0]
	u_l=find_midpt(lines[1][int(pos[0][1])])[0]
	z=b*f/(abs(u_l-u_r)+offset)/ps

	print 'the first distance estimation is '+str(z)+' m' 

	return z

#------------------------------------------------
# input: u_r is (u,v) np.array for the right camera, b - baselength in mm, A_r/l are the intrisc camera parameters matrices

def est_im_pt_RtoL(A_l, A_r, u_r, b=90.*10**(-3)):

	B_l=np.dot(A_l, np.c_[np.eye(3), np.zeros(3).T])
	B_r=np.dot(A_r, np.c_[np.eye(3), np.array([0, b, 0]).T  ])
	u_r=np.array([[u_r[0]], [u_r[1]], [1]])
	t=np.array([[0],[b], [0]])
	#make matrices. STYLE: Ax=b
	#contpdb.set_trace()

	u_est=np.dot(A_l, np.dot(np.linalg.inv(A_r) ,u_r )-t)# np.dot(np.linalg.inv(B_r),u_r))
	pdb.set_trace()
	l_ratio=u_est[2] #l/r
	u_est=(1./u_est[2])*u_est

	return [u_est, l_ratio]

#-----------------------------------------------
#take non zero entries of the dS and compare if the estimated point could correspond

def match_est(u_est, im, thresh):

	if np.linalg.norm(np.sum([pt, -u_est],axis=0) for pt in im)< thresh: return pt
	else: return -1.

#--------------------------------------------
#returns (x, y, z, 1) in wolrd frame and l_l, l_r as a list

def get_coord(A_l, A_r, u_r, pt, l_ratio, b=90):

	B_l=np.dot(A_l, np.c_[np.eye(3), np.zeros(3).T])
	B_r=np.dot(A_r, np.c_[np.eye(3), np.array([0], [b], [0])])

	u_l=np.c_[pt,1.]
	u_r=np.c_[u_r,1.]

	X=np.dot(B_l-B_r,(l_ratio*u_l-u_r))
	l_r=1./X[3]
	X=X*l_r

	l=[l_ratio*l_r, l_r]
	return X

#---------------------------------------------------
#x_w is a list of lists where each el contains the coord (x,y,z,1) and the nr to which contour sequence the pt belongs.
#no return

def plot3D(x_w):

	x=[pt[0][0] for pt in x_w]
	y=[pt[0][1] for pt in x_w]
	z=[pt[0][2] for pt in x_w]

	fig=pylab.figure()
	ax=fig.gca(projection='3d')

	ax.plot(X,Y,Z, 'b.')
	pylab.axis('equal')
	pylab.show()
