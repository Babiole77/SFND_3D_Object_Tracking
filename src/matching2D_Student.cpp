#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include "matching2D.hpp"

using namespace std;
std::ofstream out_data("output.txt", std::ios_base::app);

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorNormType, std::string matcherType, std::string selectorType, std::string descriptorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // Since SIFT return CV_32F we cannot use Hamming distance
        if (descriptorType.compare("SIFT") == 0) 
        {
            descSource.convertTo(descSource, CV_8U);
            descRef.convertTo(descRef, CV_8U);
        }
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        double minDescDistRatio = 0.8;
        for(auto itr=knn_matches.begin(); itr!=knn_matches.end(); ++itr)
        {
            if((*itr)[0].distance < minDescDistRatio*(*itr)[1].distance)
                matches.push_back((*itr)[0]);
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        /*  SIFT is not supported in OpenCv > 3.4.2.16 
            Temporary workaround for a local machine */
        // extractor = cv::xfeatures2d::SIFT::create();
        std::cout << "SIFT is not supported in this OpenCV version";
    }
    else
    {
        out_data << "Enter a proper descriptor type";
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();

    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    out_data << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    out_data << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
int blockSize = 2;
int apertureSize = 3;
int minResponse = 100;
int k = 0.04; // Harris parameter

cv::Mat dst, dst_norm;
cv::Mat::zeros(img.size(), CV_32FC1);
cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
double maxOverlap = 0.0;
double t = (double)cv::getTickCount();

for(size_t j=0; j < dst_norm.rows; ++j)
{
    for(size_t i=0; i < dst_norm.cols; ++i)
    {
        int response = (int)dst_norm.at<float>(j,i);
        if (response > minResponse)
        {
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2*apertureSize;
            newKeyPoint.response = response;
            // perform non-maximum suppression (NMS) in local neighbourhood around new key point
            bool bOverlap = false;
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
            {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                if (kptOverlap > maxOverlap)
                {
                    bOverlap = true;
                    if (newKeyPoint.response > (*it).response)
                    {                      // if overlap is >t AND response is higher for new kpt
                        *it = newKeyPoint; // replace old key point with new one
                        break;             // quit loop over keypoints
                    }
                }
            }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
        }
    } // eof loop over cols
}     // eof loop over rows
t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
out_data << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

// visualize results
if (bVis)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Harris Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    int threshold = 50; 
    bool bNMS = true; // whether non-max suppression is applied
    double t = (double)cv::getTickCount();
    if (detectorType.compare("FAST") == 0)
        cv::FAST(img, keypoints, threshold, bNMS);
    else
    {
        cv::Ptr<cv::FeatureDetector> detector;
        if (detectorType.compare("BRISK") == 0)
            detector = cv::BRISK::create();
        else if (detectorType.compare("ORB") == 0)
            detector = cv::ORB::create();
        else if (detectorType.compare("AKAZE") == 0)
            detector = cv::AKAZE::create();
        else if (detectorType.compare("SIFT") == 0)
            /*  SIFT is not supported in OpenCv > 3.4.2.16 
            Temporary workaround for a local machine */          
            // detector = cv::xfeatures2d::SIFT::create();
            std::cout << "SIFT is not supported in this OpenCV version";
        else
            std::cout << "Enter a proper desriptor type";
        detector->detect(img, keypoints);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    out_data << detectorType << " detector has found " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}
  