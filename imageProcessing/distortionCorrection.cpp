#include <iostream>
#include <sstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

int main()
{
    // camera matrix
    double alpha = 3.0028e3;
    double beta = 2.911e3;
    double gamma = 4.9566e1;
    double u0 = 1.1689e3;
    double v0 = 9.4511e2;
    cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
    cameraMatrix.at<double>(0,0) = alpha;
    cameraMatrix.at<double>(1,1) = beta;
    cameraMatrix.at<double>(2,2) = 1;
    cameraMatrix.at<double>(0,1) = gamma;
    cameraMatrix.at<double>(0,2) = u0;
    cameraMatrix.at<double>(1,2) = v0;

    // distortion coefficients (we are only considering the radial "k" coeffs)
    double k1 = -1.57e-9;
    double k2 = 8.015e-15;
    double p1 = 0;
    double p2 = 0;
    cv::Mat distortionCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    distortionCoeffs.at<double>(0,0) = k1;
    distortionCoeffs.at<double>(1,0) = k2;
    distortionCoeffs.at<double>(2,0) = p1;
    distortionCoeffs.at<double>(3,0) = p2;

    

    int nImages = 500;
    int skip = 10;
    cv::Mat img, corrected, difference;

    // load images and apply camera matrices
    for (int frame = 0; frame < nImages; frame+=skip)
    {
        std::cout << frame/skip << std::endl;

        // image file name
        std::ostringstream name;
        name << "demo_images/";
        if (frame < 10) name << "0";
        if (frame < 100) name << "0";
        name <<  frame << ".jpg";

        // load image
        img = cv::imread(name.str(), 0);

        // apply camera matrix
        cv::undistort(img, corrected, cameraMatrix, distortionCoeffs);
        
        // get the difference
        cv::absdiff(img, corrected, difference);

        cv::pyrDown(difference, difference);
        cv::imshow("difference", difference);
        char c = cv::waitKey(30);
        if (c == 'q') break;
    }
}
