/**
 * Program opens first camera, captures 100 images, displays them, and
 * writes the images to jpg file in images/ directory.
 *
 * Based on the xiApiPlusOcvExample.cpp file from XIMEA.
 **/
#include <iostream>
#include <sstream>
#include "xiApiPlusOcv.hpp"
#include "opencv2/highgui/highgui.hpp"

int nImages = 100;

int main(int argc, char* argv[])
{
    try
    {
        // set up camera
        xiAPIplusCameraOcv cam;
        cam.OpenFirst(); // open the camera
        cam.SetExposureTime(10000); // exposure time 10 ms
        cam.StartAcquisition(); // start aquiring images
        XI_IMG_FORMAT format = cam.GetImageDataFormat(); // image format

        cv::Mat cap;
        for (int i = 0; i < nImages; i++)
        {
            cap = cam.GetNextImageOcvMat(); // capture image

            // normalize image if needed (depending on format)
            if (format == XI_RAW16 || format == XI_MONO16)
            {
                cv::normalize(cap, cap, 0, 65536, cv::NORM_MINMAX,
                        -1, cv::Mat());
            }

            // display
            cv::imshow("image", cap);
            cv::waitKey(10);

            // write jpeg
            std::ostringstream name;
            name << "images/";
            if (i < 10) name << "0";
            name << i << ".jpg";
            std::cout << "Writing image " << name.str() << std::endl;
            cv::imwrite(name.str(), cap);
        }

        // shut down the camera
        cam.StopAcquisition();
        cam.Close();
    } 
    catch(xiAPIplus_Exception& exp)
    {
        std::cerr << "Error:" << std::endl;
        exp.PrintError();
        cv::waitKey(2000);
    }
    return 0;
}
