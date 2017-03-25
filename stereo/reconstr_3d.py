#!/usr/bin/python

import numpy as np
from PIL import Image
import pylab
import pdb
#import ox
import cv2

def get_f(data):

#K is the kamera matrix
#data is the set of kp for im1 and im2

	F=[]
#1.) replace kp by coord	
	for i, group in enumerate(data):
		for j, kpts in enumerate(group):
			for kp in kpts:
				kp=np.array([kp.pt[0], kp.pt[1]])
		#pdb.set_trace()
#2.) compute the A matrix 
	for i, group in enumerate(data):
		n=len(group)
		A=np.ones((n,9))

		for j in range(n):
			x1=group[j][0].pt[0]
			y1=group[j][0].pt[1]
			x2=group[j][1].pt[0]
			y2=group[j][1].pt[1]
			A[j, :]= [x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0]

		U,S,V=np.linalg.svd(A)
		F_gr=V[-1].reshape(3,3)
		U,S,V=np.linalg.svd(F_gr)
		S[2]=0
		F_gr=np.dot(U,np.dot(np.diag(S),V))
		F.append(F_gr)

	return F

def get_e(data, K):

	E=[]
#1.) replace kp by coord	
	for i, group in enumerate(data):
		for j, kpts in enumerate(group):
			for kp in kpts:
				kp=np.array([kp.pt[0], kp.pt[1]])
		#pdb.set_trace()
#2.) compute the A matrix 
	for i, group in enumerate(data):
		n=len(group)
		A=np.ones((n,9))

		for j in range(n):
			#pdb.set_trace()
			x1=np.c_[np.array([group[j][0].pt]), 1]
			x2=np.c_[np.array([group[j][1].pt]), 1]
			
			x1_k=np.dot(np.linalg.inv(K), x1.T)
			x2_k=np.dot(np.linalg.inv(K), x2.T)
			A[j, :]= [x1_k[0]*x2_k[0], x2_k[0]*x1_k[1], x2_k[0], x2_k[1]*x1_k[0], x2_k[2]*x1_k[2], x2_k[1], x1_k[0], x1_k[1], 1.0]

		U,S,V=np.linalg.svd(A)
		E_gr=V[-1].reshape(3,3)
		U,S,V=np.linalg.svd(E_gr)
		S[2]=0
		E_gr=np.dot(U,np.dot(np.diag(S),V))
		E.append(E_gr)
		#pdb.set_trace()
	return E

def cam_matrix_f(F):
#compute second P matrix assuming P1=[I 0]

	P1=np.c_[(np.eye(3),np.zeros((3,1)))]
	#epipole 2
	U, S, V=np.linalg.svd(F.T)
	e=V[-1]
	e=e/e[2]

	Te=np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
	return P1, np.vstack((np.dot(Te,F.T).T,e)).T


def cam_matrix_e(E):

	U,S,V=np.linalg.svd(E)
	if np.linalg.det(np.dot(U,V))<0: V=-V
	E=np.dot(U, np.dot(np.diag([0,0,1.]),V))

	W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
	Z=np.array([[0,1,0],[-1,0,0],[0,0,0]])

	P2=(np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,         np.vstack(( np.dot(U,np.dot(W,V)).T,-U[:,2])).T,    np.vstack((  np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,       np.vstack((   np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T  )
	#pdb.set_trace()
	P1=np.c_[(np.eye(3),np.zeros((3,1)))]

	return P1, P2

def find_P2(E, P1, P2,data, K):
#P2 has 4 possibilites: choose the one with the most inliers

	ind=0
	maxres=0
	for j in range(4):
		for kpts in data:
			X=triang_pts(P1, P2[j], kpts, K)
			d1=np.dot(P1, X)[2]
			d2=np.dot(P2[j], X)[2]
			#pdb.set_trace()
			if d1+d2>maxres:
				maxres=d1+d2
				ind=j
	return P2[ind]



def triang_pts(P1, P2, kpts, K=np.eye(3)):

	x1=np.dot(np.linalg.inv(K), np.c_[np.array([kpts[0].pt]),1].T)
	x2=np.dot(np.linalg.inv(K), np.c_[np.array([kpts[1].pt]),1].T)

	M=np.zeros((6,6))
	M[:3,:4]=P1
	M[3:,:4]=P2
	M[:3,4]=[-x1[0],-x1[1],-x1[2]]
	M[3:,5]=[-x2[0],-x2[1],-x2[2]]

	U,S,V=np.linalg.svd(M)
	X=V[-1,:4]
	return X/X[3]
