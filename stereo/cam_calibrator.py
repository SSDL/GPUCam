#!/usr/bin/python

import rospy
import sensor_msgs

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:
  def __init__(self):
    self.calib_flags = 0
    self.pattern = Patterns.Chessboard

  def loadImages(self, cal_img_path, name, n_corners, square_length, display_flag):
    self.name = name
    self.cal_img_path = cal_img_path

    self.boards = []
    self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
    self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

    if display_flag:
      fig = plt.figure('Corner Extraction', figsize = (12,5))
      gs = gridspec.GridSpec(1,2)
      gs.update(wspace=0.025, hspace=0.05)

    for i, file in enumerate(os.listdir(self.cal_img_path)):
      #print file
      img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
      img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
      drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

      if display_flag:
        ax = plt.subplot(gs[0,0])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        ax = plt.subplot(gs[0,1])
        plt.imshow(drawable.scrib)
        plt.axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

        plt.show(block=False)
        plt.waitforbuttonpress()

    # Useful parameters
    self.d_square = square_length                             # Length of a chessboard square
    self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
    self.n_chessboards = len(self.c.good_corners)             # Number of examined images
    self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
    self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

  def undistortImages(self, A, k, scale = 0):
    Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
    mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
    Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
    mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

    if k[0] != 0:
      n_plots = 3
    else:
      n_plots = 2

    fig = plt.figure('Image Correction', figsize = (6*n_plots, 5))
    gs = gridspec.GridSpec(1, n_plots)
    gs.update(wspace=0.025, hspace=0.05)

    for i, file in enumerate(os.listdir(self.cal_img_path)):
      img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
      img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
      img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

      ax = plt.subplot(gs[0,0])
      ax.imshow(img_dist, cmap='gray')
      ax.axis('off')

      ax = plt.subplot(gs[0,1])
      ax.imshow(img_undist_no_k, cmap='gray')
      ax.axis('off')

      if k[0] != 0:
        ax = plt.subplot(gs[0,2])
        ax.imshow(img_undist_w_k, cmap='gray')
        ax.axis('off')

      plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
      fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

      plt.show(block=False)
      plt.waitforbuttonpress()

  def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, k):# = np.zeros(2)):
    # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

    fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize = (8,6))
    plt.clf()

    for p in range(self.n_chessboards):
      plt.clf()
      ax = plt.subplot(111)
      ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
      u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
      ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

      box = ax.get_position()
      ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
      if k[0] != 0:
        u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
        ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

      ax.axis([0, self.w_pixels, 0, self.h_pixels])
      plt.gca().set_aspect('equal', adjustable='box')
      plt.title('Chessboard {0}'.format(p+1))
      ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

      plt.show(block=False)
      plt.waitforbuttonpress()

  def plotBoardLocations(self, X, Y, R, t):
    # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

    ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1),]
    s_cam = 0.02
    d_cam = 0.1
    xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
               [0, -s_cam, -s_cam, s_cam, s_cam],
               [0, -d_cam, -d_cam, -d_cam, -d_cam]]
    ind_cam = [[0,1,2],[0,2,3],[0,3,4],[0,4,1]]
    verts_cam = []
    for i in range(len(ind_cam)):
      verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                            [xyz_cam[1][j] for j in ind_cam[i]],
                            [xyz_cam[2][j] for j in ind_cam[i]])])

    fig = plt.figure('Estimated Chessboard Locations', figsize = (12,5))
    axim = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    boards = []
    verts = []
    for p in range(self.n_chessboards):

      M = []
      W = np.column_stack((R[p],t[p]))
      for i in range(4):
        M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
        M_tld *= np.sign(M_tld[2])
        M_tld[2] *= -1
        M.append(M_tld[0:3])

      M = (np.array(M).T).tolist()
      verts.append([zip(M[0],M[1],M[2])])
      boards.append(Poly3DCollection(verts[p]))

    for i, file in enumerate(os.listdir(self.cal_img_path)):

      img = cv2.imread(self.cal_img_path + '/' + file, 0)
      axim.imshow(img, cmap='gray')
      axim.axis('off')

      ax3d.clear()

      for j in range(len(ind_cam)):
        cam = Poly3DCollection(verts_cam[j])
        cam.set_color('green')
        cam.set_alpha(0.2)
        ax3d.add_collection3d(cam)
        
      for p in range(self.n_chessboards):
        if p == i:
          boards[p].set_color('blue')
          boards[p].set_alpha(1.0)
        else:
          boards[p].set_color('red')
          boards[p].set_alpha(0.1)

        ax3d.add_collection3d(boards[p])
        ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
        plt.show(block=False)

      view_max = 0.2
      ax3d.set_xlim(-view_max,view_max)
      ax3d.set_ylim(-view_max,view_max)
      ax3d.set_zlim(-5*view_max,0)
      ax3d.set_xlabel('X axis')
      ax3d.set_ylabel('Y axis')
      ax3d.set_zlabel('Z axis')

      plt.tight_layout()
      fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

      plt.show(block=False)

      raw_input('<Hit Enter To Continue>')

  def writeCalibrationYaml(self, A, k):
    self.c.intrinsics = A
    self.c.distortion = np.hstack((k, np.zeros(3))).reshape((5,1))
    self.c.name = self.name
    self.c.R = np.eye(3)
    self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
    self.c.size = [self.w_pixels, self.h_pixels]

    filename = self.name + '_calibration.yaml'
    with open(filename, 'w') as f:
      f.write(self.c.yaml())

    print('Calibration exported successfully to ' + filename)

  def getMeasuredPixImageCoord(self):
    u_meas = []
    v_meas = []
    for chessboards in self.c.good_corners:
      u_meas.append(chessboards[0][:,0][:,0])
      v_meas.append(self.h_pixels - chessboards[0][:,0][:,1])   # Flip Y-axis to traditional direction

    return u_meas, v_meas   # Lists of arrays (one per chessboard)

  def genCornerCoordinates(self, u_meas, v_meas):
    x=np.linspace(0, 180, num=8) #instead of 9
    x=x.T
    y=np.linspace(0, 135, num=6) #instead of 7
    y=y.T
    X_l=np.hstack([x,x,x,x,x,x]) # one more x
    Y=np.c_[y,y,y,y,y,y,y,y] #one more y
    Y_l=np.hstack([Y[0,:],Y[1,:],Y[2,:],Y[3,:], Y[4,:],Y[5,:]])#,Y[6,:]])
    X=[]
    Y=[]
    for i in range(len(u_meas)):
        X.append(X_l)
        Y.append(Y_l)
    return X, Y


  def estimateHomography(self, u_meas, v_meas, XX, YY):
    n=6*8#n=7*9
    L=np.zeros((2*n, 9))
    #M_tilde(i,j)=np.array([X[i,j], Y[i,j], 1])
    for i in range(n-1):
        u=u_meas[i]
        v=v_meas[i]
        X=XX[i]
        Y=YY[i]
        a=np.array([X, Y, 1, 0,0,0, -u*X, -u*Y, -u])
        b=np.array([0,0,0, X, Y, 1, -v*X, -v*Y, -v])
        L[2*i,:]=a
        L[2*i+1,:]=b
    #x=np.linalg.solve(L,np.zeros(2*n).T) - NOPE!
    U, s, V= np.linalg.svd(L)
    sig=np.argmin(s)
    x=V[sig,:]#/s[sig] # cause it is V.T that we compute with svd func

    H=np.vstack([x[:3], x[3:6], x[6:]])
    return H

  def getCameraIntrinsics(self, h):
    #pdb.set_trace()
    def get_v(i,j, H):
        v= np.array([H[0,i]*H[0,j], H[0,i]*H[1,j]+H[1,i]*H[0,j], H[1,i]*H[1,j], H[2,i]*H[0,j]+H[0,i]*H[2,j], H[2,i]*H[1,j]+H[1,i]*H[2,j], H[2,i]*H[2,j]])
        #pdb.set_trace()
        return v
    V=np.zeros((1,6))
    for i in range(len(h)):
        H=h[i]
        v_12= get_v(0,1, H)
        v_11= get_v(0,0, H)
        v_22= get_v(1,1, H)
    
        v=np.vstack([v_12.T, (v_11.T-v_22.T)])
        V=np.vstack([V,v])
    V=V[1:,:]
    
    U,s,W= np.linalg.svd(V)
    s_arg=np.argmin(s)
    b= W[s_arg,:]#/s[s_arg]

    #b=np.linalg.solve(V, np.zeros(2*7*9).T) - NOPE!
    B= np.array([[b[0], b[1], b[2]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    
    #compute intrinsic parameters
    v_0=(b[1]*b[3] - b[0]*b[4])/(b[0]*b[2]-b[1]**2.0)
    l= b[5] - (b[3]**2.0 + v_0*(b[1]*b[3]- b[0]*b[4])) /b[0]
    alpha=np.sqrt(abs(l/b[0]))
    beta= np.sqrt(abs(l*b[0]/(b[0]*b[2]-b[1]**2.0)))
    gamma=-b[1]*alpha**2.0*beta/l
    u_0= gamma*v_0/alpha - b[3]*alpha**2.0/l

    #define A
    #pdb.set_trace()
    A=np.array([[alpha, gamma, u_0],[0, beta, v_0],[0,0,1]])
    return A

  def getExtrinsics(self, H, A):
    ah=np.dot(np.linalg.inv(A),H)
    L=1/np.sqrt(ah[0,0]**2.0 +ah[1,0]**2.0+ah[2,0]**2.0)
    r1=L*ah[:,0]
    r2=L*ah[:,1]
    r3=np.cross(r1, r2)
    t=L*ah[:,2]
    R=np.c_[r1,r2,r3]
    #U, s, V = np.linalg.svd(ah)
    #s=np.min(s)
    #rt=(1/s)*ah
    #print size(rt)#is it 3*3?
    return R, t

  def transformWorld2NormImageUndist(self, XX, YY, Z, R, t):
    x=np.array([0])
    y=np.array([0])
    #r=np.zeros((3,1))
    #R=np.c_[R,r]
    for i in range(np.size(XX)):
        X=XX[i]
        Y=YY[i]
        x_w=np.array([X, Y, 1]).T
        x_rot=np.dot(R,x_w)
        x=np.vstack([x,x_rot[0]/x_rot[2]+t[0]])
        y=np.vstack([y,x_rot[1]/x_rot[2]+t[1]])
    x=x[1:]
    y=y[1:]
    return x, y

  def transformWorld2PixImageUndist(self, XX, YY, Z, R, t, A):
    u=np.array([0])
    v=np.array([0])
    rt=np.c_[R[:,0], R[:,1],t]
    for i in range(np.size(XX)):
        X=XX[i]
        Y=YY[i]
        x_w=np.array([X, Y, 1]).T
        u_w=np.dot(np.dot(A,rt),x_w)
        v=np.vstack([v, u_w[1]/u_w[2]]) # u[2] is the scaling factor
        u=np.vstack([u, u_w[0]/u_w[2]])
    u=u[1:]
    v=v[1:]
    #pdb.set_trace()
    return u, v

  def transformWorld2NormImageDist(self, X, Y, Z, R, t, k):
    #x,y=transformWorld2NormImageUndist( X, Y, np.ones(np.size(X)), R, t, A)
    x=np.array([0])
    y=np.array([0])
    for i in range(np.size(X)):
        Xi=X[i]
        Yi=Y[i]
        x_w=np.array([Xi, Yi, 1]).T
        x_rot=np.dot(R,x_w)
        x=np.vstack([x,x_rot[0]/x_rot[2]+t[0]])
        y=np.vstack([y,x_rot[1]/x_rot[2]+t[1]])
    x=x[1:]
    y=y[1:]
    x_d=np.array([0])
    y_d=np.array([0])
    for i in range(np.size(X)):
        r=np.sqrt(x[i]**2.0+y[i]**2.0)
        x_d=np.vstack([u_d,(x[i]+ x[i]*(k[0]*r+k[1]*r**2.0))])
        y_d=np.vstack([v_d,(y[i]+ y[i]*(k[0]*r+k[1]*r**2.0))])
    x_d=x_d[1:,:]
    y_d=y_d[1:,:]
    return x_d, y_d

  def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):
    u_0=A[0,2]
    v_0=A[1,2]
    rt=np.c_[R[:,0], R[:,1],t]
    x=np.array([0])
    y=np.array([0])
    u=np.array([0])
    v=np.array([0])
    for i in range(np.size(X)):
        x_w=np.array([X[i], Y[i], 1]).T
        u_w=np.dot(np.dot(A,rt),x_w)
        v=np.vstack([v, u_w[1]/u_w[2]]) # u[2] is the scaling factor
        u=np.vstack([u, u_w[0]/u_w[2]])
        x_rot=np.dot(R,x_w)
        x=np.vstack([x,x_rot[0]/x_rot[2]+t[0]])
        y=np.vstack([y,x_rot[1]/x_rot[2]+t[1]])
    x=x[1:]
    y=y[1:]
    u=u[1:]
    v=v[1:]

    u_d=np.array([0])
    v_d=np.array([0])
    for i in range(np.size(X)):
        r=np.sqrt(x[i]**2.0+y[i]**2.0)
        u_d=np.vstack([u_d,(u[i]+ (u[i]-u_0)*(k[0]*r+k[1]*r**2.0))])
        v_d=np.vstack([v_d,(v[i]+ (v[i]-v_0)*(k[0]*r+k[1]*r**2.0))])
    u_d=u_d[1:,:]
    v_d=v_d[1:,:]
    return u_d, v_d

  def estimateLensDistortion(self, u_m, v_m, XX, YY, RR, tt, A):
    u_0 =A[0,2]
    v_0=A[1,2]
    D=np.zeros((1,2))
    d=np.array([0])
    for j in range(len(u_m)):
        u_meas=u_m[j]
        v_meas=v_m[j]
        X=XX[j]
        Y=YY[j]
        R=RR[j]
        t=tt[j]
        #x,y=transformWorld2NormImageUndist( X, Y, np.ones(np.size(X)), R, t, A)
        x=np.array([0])
        y=np.array([0])
        for i in range(np.size(X)):
            Xi=X[i]
            Yi=Y[i]
            x_w=np.array([Xi, Yi, 1]).T
            x_rot=np.dot(R,x_w)
            x=np.vstack([x,x_rot[0]/x_rot[2]+t[0]])
            y=np.vstack([y,x_rot[1]/x_rot[2]+t[1]])
        x=x[1:]
        y=y[1:]

        #u,v=transformWorld2PixImageUndist(X, Y, np.ones(np.size(X)), R, t, A)
        u=np.array([0])
        v=np.array([0])
        rt=np.c_[R[:,0], R[:,1],t]
        for i in range(np.size(X)):
            Xi=X[i]
            Yi=Y[i]
            x_w=np.array([Xi, Yi, 1]).T
            u_w=np.dot(np.dot(A,rt),x_w)
            v=np.vstack([v, u_w[1]/u_w[2]]) # u[2] is the scaling factor
            u=np.vstack([u, u_w[0]/u_w[2]])
        u=u[1:]
        v=v[1:]


        D_img=np.zeros((1,2))
        d_img=np.array([0])
        for i in range(np.size(u_meas)):
            d_i=np.array([u_meas[i]-u[i], v_meas[i]-v[i]])
            r=x[i]**2.0+y[i]**2.0
            D_i1=np.array([r*(u[i]-u_0), (u[i]-u_0)*r**2.0])
            D_i2=np.array([r*(v[i]-v_0), (v[i]-v_0)*r**2.0])
            #pdb.set_trace()
            D_img=np.vstack([D_img,D_i1.T, D_i2.T])
            d_img=np.vstack([d_img,d_i])
        D_img=D_img[1:,:]
        d_img=d_img[1:,:]
        D=np.vstack([D, D_img])
        d=np.vstack([d,d_img])
    D=D[1:,:]
    d=d[1:,:]
    k=np.dot(np.dot(np.linalg.inv(np.dot(D.T, D)),D.T),d)
    return k



