/**
 * Helper functions for using multiple cameras. Perform set up, capture,
 * and shutdown neatly. 
 **/
#include <iostream>
#include <string>
#include "xiApiPlusOcv.hpp"
#include "opencv2/highgui/highgui.hpp"

/**
 * Initialize the cameras.
 * Inputs:
 *  nCameras: how many cameras (int)
 *  camera: array of xiAPIplusCameraOcv objects
 *  format: array of formats (will be filled by the function)
 *  cameraID: array of camera ID strings
 *  exposureTime: array of exposure times (int, in microseconds)
 *  bandwidthLimit: array of bandwidth limits (int, in Mbit per second)
 **/
void initCameras(int nCameras, xiAPIplusCameraOcv *camera,
        XI_IMG_FORMAT *format, std::string *cameraID, int *exposureTime,
        int *bandwidthLimit)
{
    for (int c = 0; c < nCameras; c++)
    {
        std::cout << "Opening camera \"" << cameraID[c] << "\"" << std::endl;
        camera[c].OpenBySN(&cameraID[c][0u]); // open camera by serial number
        camera[c].SetExposureTime(exposureTime[c]); // set exposure time
        camera[c].SetBandwidthLimit(bandwidthLimit[c]); // set bandwidth limit
        camera[c].StartAcquisition(); // start getting images

        // get format
        format[c] = camera[c].GetImageDataFormat();
    }
}

/**
 * Get an image from each camera and normalize if necessary.
 * Inputs:
 *  nCameras: how many cameras (int)
 *  camera: array of xiAPIplusCameraOcv objects
 *  cap: array of cv Mat objects for putting images in 
 *  format: array of XI_IMG_FORMAT
 **/
void captureNextImage(int nCameras, xiAPIplusCameraOcv *camera, cv::Mat *cap,
        XI_IMG_FORMAT *format)
{
    for (int c = 0; c < nCameras; c++)
    {
        // get image
        cap[c] = camera[c].GetNextImageOcvMat().clone();

        // normalize if required
        if (format[c] == XI_RAW16 || format[c] == XI_MONO16)
        {
            cv::normalize(cap[c], cap[c], 0, 65536, cv::NORM_MINMAX,
                    -1, cv::Mat());
        }
    }
}

/**
 * Shuts down the cameras
 * Input:
 *  nCameras: how many cameras (int)
 *  camera: array of xiAPIplusCamerOcv objects
 **/
void closeCameras(int nCameras, xiAPIplusCameraOcv *camera)
{
    for (int c = 0; c < nCameras; c++)
    {
        camera[c].StopAcquisition();
        camera[c].Close();
    }
}
