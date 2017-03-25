#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pdb

from cam_calibrator import CameraCalibrator


def main(cal_img_path):
  cc = CameraCalibrator()
  name = 'webcam'             
  n_corners=[6,8] 
  square_length = 0.02978        # Chessboard square length in meters

  display_flag = False
  cc.loadImages(cal_img_path, name, n_corners, square_length, display_flag)

  u_meas, v_meas = cc.getMeasuredPixImageCoord()
  X, Y= cc.genCornerCoordinates(u_meas, v_meas)
  H=[]
  R_final=[]
  t_final=[]
  V=np.array([0])
  for i in range(len(u_meas)):
      u_m=u_meas[i]
      v_m=v_meas[i]
      XX=X[i]
      YY=Y[i]

      h= cc.estimateHomography(u_m, v_m, XX, YY)
      H.append(h)

  A=cc.getCameraIntrinsics(H)
  print A
  for i in range(len(u_meas)):
      u_m=u_meas[i]
      v_m=v_meas[i]
      XX=X[i]
      YY=Y[i]
      h=H[i]
      R, t=cc.getExtrinsics(h, A)
      R_final.append(R)
      t_final.append(t)
      x,y = cc.transformWorld2NormImageUndist(XX, YY, np.ones(np.size(XX)), R, t)
      u,v = cc.transformWorld2PixImageUndist(XX, YY, np.ones(np.size(XX)), R, t, A)

  k= cc.estimateLensDistortion(u_meas, v_meas, X, Y, R_final, t_final, A)
  print k
  
  #--------------------------------------------------------
  #make executable if you may wnat to see pics
 
  #cc.plotBoardPixImages(u_meas, v_meas, X, Y, R_final, t_final, A, k)
  #k=k.flatten()
  #cc.undistortImages(A,k)
  #cc.writeCalibrationYaml(A, k)

  return A, k



if __name__ == '__main__':
    im_list=['./chess_L','./chess_R']
    for chess in im_list:
        try:
            A,k= main(chess)
        except Exception as e:
            import traceback
            traceback.print_exc()
