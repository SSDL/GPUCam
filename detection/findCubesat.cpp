#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

inline double pow2(double a) {return a * a;};

/**
 * Get the center of mass and principle axes of a binary image
 * Inputs:
 *  edge: Mat of edges (output from Canny)
 *  cm: Point2f to put the cm in
 *  axes: Size to put semi-major and semi-minor axes in
 *  theta: double to put angle (in degrees) into
 **/
void getPrincipleAxes(cv::Mat &edge, cv::Point2f &cm, cv::Size &axes,
        double &theta)
{
    // get the moments
    cv::Moments mu = cv::moments(edge, true);

    // center of mass
    cm = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


    // compute eigenvalues
    double lambda1 = 0.5 * ((mu.mu20 + mu.mu02) + sqrt(4.0*pow2(mu.mu11) + 
                pow2(mu.mu20 - mu.mu02)));
    double lambda2 = 0.5 * ((mu.mu20 + mu.mu02) - sqrt(4.0*pow2(mu.mu11) + 
                pow2(mu.mu20 - mu.mu02)));

    // get the principle axis lengths
    double semimajor = 2.0 * sqrt(lambda1 / mu.m00);
    double semiminor = 2.0 * sqrt(lambda2 / mu.m00);
    axes = cv::Size(semimajor, semiminor);

    // orientation
    theta = 0.5 * atan(mu.mu11 / (mu.mu20 - mu.mu02)) * 180.0 / M_PI;
}

/**
 * Mask an image by a rotated ellipse, creating a new Mat that is zero outside
 * the ellipse region.
 * Inputs:
 *  input: Mat to be masked
 *  output: Mat for output of mask
 *  cm: Point2f with center of ellipse
 *  axes: Size with semimajor and semiminor axes
 *  scale: double reperesenting how much to scale the ellipse for masking
 **/
void maskByEllipse(cv::Mat &input, cv::Mat &output, cv::Point2f &cm,
        cv::Size &axes, double theta, double scale)
{
    // draw ellipse on empty Mat for mask
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::ellipse(mask, cm, cv::Size(axes.width * scale, axes.height * scale),
            theta, 0, 360, 255, -1);

    // apply mask
    output = cv::Mat::zeros(input.size(), input.type());
    input.copyTo(output, mask);
}

/**
 * Satellite detection code. Performs canny filter, finds CM of edges,
 * attempts to remove extraneous edges, and returns the computed location.
 * Inputs:
 *  img: Mat with the image being processed
 *  threshold: double, fraction of image width the semimajor axis is allowed to
 *             before the satellite is considered not detected
 **/
cv::Point2f detect(cv::Mat &img, double threshold)
{
    cv::Mat blur, thresh, edge, edge2, xGrad, yGrad;
    cv::Size axes;
    cv::Point2f cm;
    double theta;

    // properties for Gaussian blur and Canny
    cv::Size kernel(5,5);
    double std = 1.4;

    // edge detection with aperture 3 for sobel operator
    // apply gaussian filter
    cv::GaussianBlur(img, blur, kernel, std, std);

    // threshold using Otsu's method
    double high = cv::threshold(img, thresh, 0, 255,
            cv::THRESH_BINARY + cv::THRESH_OTSU);
    double low = 0.5 * high;

    // canny edge detection
    cv::Canny(blur, edge, low, high, 3);

    // get principle axes and center of mass, and mask, twice
    getPrincipleAxes(edge, cm, axes, theta);
    for (int i = 0; i < 2 && axes.width > 0 && axes.height > 0; i++)
    {
        maskByEllipse(edge, edge2, cm, axes, theta, 1.25);
        getPrincipleAxes(edge2, cm, axes, theta);
        cv::swap(edge, edge2);
    }
    if (axes.width <= 0 || axes.height <=0)
    {
        std::cout << "failed to detect an object" << std::endl;
        return cv::Point2f(-1,-1);
    }
    if (axes.width > 0.35 * img.cols || axes.height > 0.35 * img.cols)
    {
        std::cout << "failed to detect an object with confidence" << std::endl;
        return cv::Point2f(-1,-1);
    }
    return cm;
}

int main()
{
    int nImages = 500;
    cv::Mat img;

    for (int frame = 0; frame < nImages; frame++)
    {
        std::cout << frame << std::endl;

        // image file name
        std::ostringstream name;
        name << "recording2/";
        if (frame < 10) name << "0";
        if (frame < 100) name << "0";
        name <<  frame << ".jpg";

        // load image
        img = cv::imread(name.str(), 0);

        // detection
        cv::Point2f loc = detect(img, 0.35);
        
        // draw CM point if detected
        if (loc.x < 0) continue;
        cv::circle(img, loc, 12, 128, -1);
        cv::pyrDown(img, img);
        cv::imshow("contours", img);
        char c = cv::waitKey(30);
        if (c == 'q') break;
    }

}
