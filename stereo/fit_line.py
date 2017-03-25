import numpy as np
#from PlotFunctions import *
import matplotlib.pyplot as plt
import pdb

#-----------------------------------------------------------
# FindSplit
#
# This function takes in a line segment and outputs the best
# index at which to split the segment
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#         alpha - 'alpha' of input line segment (1 number)
#             r - 'r' of input line segment (1 number)
#        params - dictionary of parameters
#
# OUTPUT: SplitIdx - idx at which to split line (return -1 if
#                    it cannot be split)

def FindSplit(theta, rho, alpha, r, params):

  ##### TO DO #####
  # Implement a function to find the split index (if one exists)
  # It should compute the distance of each point to the line.
  # The index to split at is the one with the maximum distance 
  # value that exceeds 'LINE_POINT_DIST_THRESHOLD', and also does
  # not divide into segments smaller than 'MIN_POINTS_PER_SEGMENT'
  # return -1 if no split is possible
  #params = {'MIN_SEG_LENGTH': 0.00,
  #          'LINE_POINT_DIST': 20,
  #          'MIN_POINTS_PER_SEGMENT': 4, 'MAX_P2P_DIST':400}
  #################

  #line equation-> ax+by+c=0
  a=np.cos(alpha)
  b=np.sin(alpha)
  c=-r
  
  d=np.array([0])
  for i in range(len(rho)):
  #point i coordinates
      x = rho[i]*np.cos(theta[i])
      y = rho[i]*np.sin(theta[i])

  #distance
      #d_i= abs(a*x+y*b+c)/np.sqrt(a**2.0+b**2.0)
      d_i=abs(rho[i]*np.cos(theta[i]-alpha)-r)
      d=np.vstack([d,d_i])
  d=d[1:]
  #pdb.set_trace()
  for i in range(int(params['MIN_POINTS_PER_SEG'])+1):
      d[i]=0
      d[(-i)]=0
  s=np.argmax(d)
  #print d
  #print s
  
  #print s
  if d[s]>params['LINE_POINT_DIST']: #and len(d[:s+1])>=param[1] and len(d[s+1:])>=param[1]:
      splitIdx=s
  else:
      splitIdx=-1

  return splitIdx



#-----------------------------------------------------------
# FitLine
#
# This function outputs a best fit line to a segment of range
# data, expressed in polar form (alpha, r)
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#
# OUTPUT: alpha - 'alpha' of best fit for range data (1 number) (rads)
#             r - 'r' of best fit for range data (1 number) (m)

def FitLine(theta, rho):

  n=len(theta)
  if n==0: return

  sum_1=0
  for i in range(n):
      val=rho[i]**2.0*np.sin(2*theta[i])
      sum_1=sum_1+val
  sum_2=0
  for i in range(n):
      sum_j=0
      for j in range(n):
           sum_j=sum_j+rho[j]*rho[i]*np.cos(theta[i])*np.sin(theta[j])
      sum_2=sum_2+sum_j

  sum_4=0
  for i in range(n):
      val=rho[i]**2.0*np.cos(2*theta[i])
      sum_4=sum_4+val
  sum_5=0
  for i in range(n):
      sum_j=0
      for j in range(n):
           sum_j=sum_j+rho[j]*rho[i]*np.cos(theta[i]+theta[j])
      sum_5=sum_5+sum_j

  v=(sum_4 - sum_5/n)
  w=(sum_1-(2.0/n)*sum_2)
  #pdb.set_trace()
  alpha= (0.5)*np.arctan2(w,v)+(np.pi/2.0)

  sum_3=0
  for i in range(n):
      sum_3=sum_3+rho[i]*np.cos(theta[i]-alpha)

  r=(1.0/n)*sum_3
  if r<0:
      r=abs(r)
      alpha=alpha-np.pi
  
  #alpha= alpha%(2.0*np.pi)


  return alpha, r

#------------------------------------------------------------------
# Convert to polar coordinates
# Input: x,y
# Output: alpha, r

def convert_to_polar(coord):
	coord_pol=[]#np.zeros(coord_xy)
	for i in range(len(coord)):
		x = coord[i][0]
		r=np.linalg.norm(x)
		alpha=np.arctan2(x[1],x[0])
		coord_pol.append([alpha, r, int(i)])
	return np.array(coord_pol)


#---------------------------------------------
#converts angle in rad into (-pi, pi)
def wrapToPi(a):
	b = a
	if a< -np.pi or a> np.pi:
		b = ((a+np.pi) % (2*np.pi)) - np.pi
	return b








