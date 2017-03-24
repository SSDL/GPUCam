/**
 * Program opens three cameras,  captures 100 images, displays them, and
 * writes the images to jpg files in images/ directory.
 *
 * Uses ararys to manage camera parameters.
 *
 * Can be easily extended to more (or fewer) cameras by adding more (or
 * fewer) cameras by serial number.
 **/
#include <iostream>
#include <sstream>
#include <string>
#include "xiApiPlusOcv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cameraHelper.cpp"

int nImages = 100;
int nCameras = 3;

int main(int argc, char* argv[])
{
    // cameras and format of each camera
    xiAPIplusCameraOcv camera[nCameras];
    XI_IMG_FORMAT format[nCameras];
    
    // array of camera names for saving images with nice file names
    std::string cameraName[nCameras] = {"center", "left", "right"};

    // array of how many times to downsample each camera image before display
    int downsample[nCameras] = {2, 1, 1};

    // array of camera serial numbers, up to user to know which is which
    std::string cameraID[nCameras] = {"35582542", "49607150", "49600150"};

    // array of exposure times (in microseconds)
    int exposureTime[nCameras] = {2000, 2000, 2000};

    // array of bandwidth limits (in Mbits per second)
    int bandwidthLimit[nCameras] = {2000, 600, 600};

    // array for captured images
    cv::Mat cap[nCameras];

    try
    {

        // initialize the cameras and start aquisition
        std::cout << "Initializing Cameras..." << std::endl;
        initCameras(nCameras, camera, format, cameraID, exposureTime,
                bandwidthLimit);

        // capture frames
        for (int i = 0; i < nImages; i++)
        {
            std::cout << "Image number " << i << std::endl;
            captureNextImage(nCameras, camera, cap, format);

            // save images
            for (int c = 0; c < nCameras; c++)
            {
                std::ostringstream name;
                name << "images/" << cameraName[c];
                if (i < 10) name << "0"; // padding
                name << i << ".jpg";
                cv::imwrite(name.str(), cap[c]);
            }

            // display images
            int counter = 0;
            for (int c = 0; c < nCameras; c++)
            {
                // downsampling
                for (int d = 0; d < downsample[c]; d++)
                    cv::pyrDown(cap[c], cap[c]);

                // display
                cv::imshow(cameraName[c], cap[c]);
                cv::moveWindow(cameraName[c], counter, 0);
                counter += cap[c].cols + 100;
                cv::waitKey(10);
            }
        }

        // shut down cameras
        closeCameras(nCameras, camera);
    }
    catch(xiAPIplus_Exception& exp)
    {
        std::cout << "Error:" << std::endl;
        exp.PrintError();
        cv::waitKey(2000);
        return 1;
    }


}
