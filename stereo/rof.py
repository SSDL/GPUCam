#!/usr/bin/python

from numpy import *
from PIL import Image
import pdb
import cv2

def denoise(im, U_init, tol=0.15, tau=0.125, tv_weight=100):

	#initialize
	m,n= im.shape
	U=U_init
	Px=im
	Py=im
	error=1

	#update
	while (error>tol):
		U_old=U
		gradUx=roll(U, -1, axis=1)-U
		gradUy=roll(U, -1, axis=0)-U

		Px_new=Px+(tau/tv_weight)*gradUx
		Py_new=Px+(tau/tv_weight)*gradUy

		Norm_new=maximum(1,sqrt(Px_new**2.+Py_new**2.))

		DivP=(Px-roll(Px, 1, axis=1))+(Py-roll(Py, 1, axis=0))

		U=im+tv_weight*DivP
		error=linalg.norm(U-U_old)/sqrt(n*m)

	return U, im-U
