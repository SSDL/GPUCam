#!/usr/bin/python

import numpy as np
from scipy.ndimage import filters
from PIL import Image
from pylab import *
import pdb
import os
import cv2
from indv_img import *
from pair_img import *
from reconstr_3d import *
from sat_detect import *
from fit_line import *
from build_3d import *
import random

from cal_workspace import main as cal
from rof import *


#---------------------
#im_list is [R, L] init
#image number to run the file with
im_list=['.jpg', '.jpg']
shot_nbr=39

#---------------------
#thresholds an parameters:
thresholds={'kp_circle':100, 'best_n_matches':4, 'border':5, 'patch_wid':5, 'contour_rad':80, 'intensity':70, 'est_thresh':5, 'baselen':90*10**(-3.), 'focal_len':16*10**(-3.), 'pix_size':5.3*10**(-6.)}

#----------------------

def getkey(item):
	return int(item[:-6])
def key2(item):
	return item[-6:]

path= "./pic_stream"
listing = sorted(os.listdir(path), key=getkey)

im=[]
for i in np.linspace(1, len(listing), int(len(listing)/3)):im.append(listing[int(i-1):int(i+2)]) 

for el in im[shot_nbr]:
	if el[-5] == 'R':im_list[0]=el
	if el[-5] == 'L':im_list[1]=el

#should yield in im_list=['39_R.jpg', '39_L.jpg']

#--------------------- START CODE:
patch_wid=5
thresh=0.5
desc=[]
kp_locs=[]
g=[]
kp_groups_im=[]


#cam parameters estimation, not needed if computaion is executated!
#f=2.5 # focal length
#c_x, c_y= (480.,272.)# - center of the im (often wid/2, hei/2)
#s=0 #- angle (set zero!!)
#alpha=1 #1.75x1.75 um - if pix is square : alpha =1!
#K=np.array([[alpha*f, s, c_x],[0, f, c_y],[0,0,1.0]])



#find intrinsic cam paramters and lens distortion
chess=['./chess_L','./chess_R']
A_r, k_r=cal(chess[1])
A_l, k_l=cal(chess[0])
#A_r[0,1]=wrapToPi(A_r[0,1])
#A_l[0,1]=wrapToPi(A_l[0,1])


#---------------------

for im_name in im_list:
	#read im
	im=np.array(Image.open(im_name).convert('L'), 'f')
	#im, T=denoise(im, im)
	#cv2.imwrite('test.jpg', im)

	#get kp and loc with SIFT
	gray, kp, kp_loc=get_kp(im_name, thresholds['border'])
	kp_locs.append(kp_loc)

	#filter groups: pt of iterest (idx-group is a list of idx of the poitn in te group)
	idx_groups=[]
	kp_groups=[]
	for i in range(len(kp)):
		idx=kp_circle(thresholds['kp_circle'],kp, kp_loc,i)
		if np.linalg.norm(idx)!=0:
			idx_groups.append(idx)
			kp_group=[]
			for j in idx:
				kp_group.append(kp[j])
				kp[j]=0 #set zeros - can not be in two groups!
			kp_groups.append(kp_group)
	kp_groups_im.append(kp_groups)
	

	G=print_kp(gray, kp_groups, im_name)
	g.append(gray)
	#get patches that can be compared to other pic
	desc_i=desc_image(gray, kp_groups, patch_wid) 
	desc.append(desc_i)
#compare areas of interest in pair of images: through cross correlation
#ncc_m give the sum of the ncc scores (above threshhold) divided bz the total amount of checkpoints in order to identify two similar objects in both images
#find best n- matches, pos in kp_groups, conpare data between both images, find mathcing points !


matches, ncc_m=match(desc[0], desc[1], thresh, thresholds['patch_wid'])
pos0= best_n(ncc_m, thresholds['best_n_matches'])

matches1, ncc_m1=match(desc[1], desc[0], thresh, thresholds['patch_wid'])
pos1= best_n(ncc_m1, thresholds['best_n_matches'])

pos=np.array([0.,0.])
for i in range(thresholds['best_n_matches']):
	if pos0[i][0]==pos1[i][1] and pos0[i][1]==pos1[i][0]:pos=np.vstack([pos, pos0[i]])
pos=pos[1:, :]
data=get_data(matches, pos, kp_groups_im) #return kp pairs

kp_matched=[[],[]]
for gr in data:
	kps=[[],[]]
	for pair in gr:
		kps[0].append(pair[0])
		kps[1].append(pair[1])
	kp_matched[0].append(kps[0])
	kp_matched[1].append(kps[1])

#for i in [0,1]:	print_kp(g[i],kp_matched[i], im_list[i])
#comment: PB is that manz keypoint are refrenced to the smae keypoint in the other image

kp_gr_filtered=[]
lines=[]
desc2=[]
for k, im_name in enumerate(im_list):
	kp_gr=[]
	for i, kp_gr_i in enumerate(kp_groups_im[k]):
		for gr in pos[:,k]:
			if i==gr: kp_gr.append(kp_gr_i)
	kp_gr_filtered.append(kp_gr)
	lines_k, contour=contour2(g[k], kp_gr_filtered[k], int(k), thresholds['contour_rad'])
	lines.append(contour)

	desc2_i=desc_im2(g[k], contour, thresholds['patch_wid'])#random.sample(contour, int((.25)*len(contour))
	desc2.append(desc2_i)
	#imshow(g[0])
	#for line in [contour[0], contour[23]]:
	#	for pt in line:
	#		plot(pt[0][0], pt[0][1],'g*')
	#show()

#unhashtag!just lots of comp
matches, ncc_m=match(desc2[0], desc2[1], thresh, thresholds['patch_wid'])
pos= best_n(ncc_m, thresholds['best_n_matches'])
print pos

#estimate distance of object
z=first_dist_est(pos, lines, thresholds['baselen'],thresholds['focal_len'],thresholds['pix_size'])#, abs(A_r[0,2]-A_l[0,2]))

#update the lines
filtered=-np.ones((1,2))
for i, el in enumerate(pos):
	if el[0] not in filtered[:,0] and el[1] not in filtered[:,1]: filtered=np.vstack((filtered, pos[i]))
pos=filtered[1:]

a=[]
b=[]
for i,j in zip(pos[:,0], pos[:, 1]):
	a.append(lines[0][int(i)])
	b.append(lines[1][int(j)])
lines[0]=a
lines[1]=b

#pdb.set_trace()

#imshow(g[0])
#for line in lines[0]:
#	for pt in line:
#		plot(pt[0][0], pt[0][1],'g*')
#show()


lines[0]=linefitting(lines[0])
lines[1]=linefitting(lines[1])

desc2=[]
for k, im_name in enumerate(im_list):
	desc2_i=desc_im2(g[k], lines[k], thresholds['patch_wid'])#random.sample(contour, int((.25)*len(contour))
	desc2.append(desc2_i)

matches, ncc_m=match(desc2[0], desc2[1], thresh, thresholds['patch_wid'])
pos= best_n(ncc_m,-1+min(len(lines[0]), len(lines[1])))

#update the lines
filtered=-np.ones((1,2))
for i, el in enumerate(pos):
	if el[0] not in filtered[:,0] and el[1] not in filtered[:,1]: filtered=np.vstack((filtered, pos[i]))
pos=filtered[1:]

a=[]
b=[]
for i,j in zip(pos[:,0], pos[:, 1]):
	a.append(lines[0][int(i)])
	b.append(lines[1][int(j)])
lines[0]=a
lines[1]=b

print pos

feat=[]
reduced=lines
for k, im in enumerate(lines):
	for i, line in enumerate(im):
		reduced[k][i]=find_keypoints(line) #either midpoint or endpoint (mean cood, standrad dev)

imshow(g[0])
for line in lines[0]:
	for pt in line:
		plot(pt[0][0], pt[0][1],'g*')
show()
pdb.set_trace()

#############-------------------------------- DEBUGGED UNTIL HERE
#ideas on how to get started with the files.

#estimate point pos in L image:

for line in lines[0]:
	u_est=[est_im_pt_RtoL(A_l, A_r, u_r)[0] for u_r in line]
	matches=[match_est(list(u_est), line, thresholds['est_thresh']) for line in lines[1]]



#--------------------using reconstr_3d

#triangulation
F_list=get_f(data)
E_list=get_e(data,K)

length_e=len(E_list)

pts_3d=[]


for i, E in enumerate(E_list):
	P1, P2=cam_matrix_e(E)
	P2=find_P2(E,P1, P2, data[i], K)

	pts_3d_gr=[]
	for kpts in data[i]:
		X=triang_pts(P1, P2, kpts, K)
		pts_3d_gr.append(X)
	pts_3d.append(pts_3d_gr)

XX=np.array([0])
YY=np.array([0])
ZZ=np.array([0])
for pt in pts_3d[1]:
	x=pt[0]
	y=pt[1]
	z=pt[2]
	XX=np.hstack([XX,x])
	YY=np.hstack([YY,y])
	ZZ=np.hstack([ZZ,z])

X=XX[1:]
Y=YY[1:]
Z=ZZ[1:]

from mpl_toolkits.mplot3d import axes3d
fig=pylab.figure()
ax=fig.gca(projection='3d')

ax.plot(X,Y,Z, 'r.')
pylab.axis('equal')
pylab.show()


