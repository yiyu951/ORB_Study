#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <string>


#include "ORBextractor.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;
using namespace ORB_SLAM3;

int main() {
    FileStorage fSetting{"/home/zhonglingjun/GXT/ORB_Study/Configs/camera.yaml", FileStorage::READ};
    vector<float> vCalibrationLeft, vCalibrationRight;
    vector<float> vPinHoleDistorsionLeft, vPinHoleDistorsionRight;
    {
        float fx = fSetting["Camera1.fx"];
        float fy = fSetting["Camera1.fy"];
        float cx = fSetting["Camera1.cx"];
        float cy = fSetting["Camera1.cy"];
        vCalibrationLeft = {fx, fy, cx, cy};
        float k1 = fSetting["Camera1.k1"];
        float k2 = fSetting["Camera1.k2"];
        float p1 = fSetting["Camera1.p1"];
        float p2 = fSetting["Camera1.p2"];
        vPinHoleDistorsionLeft = {k1, k2, p1, p2};
    }
    {
        float fx = fSetting["Camera2.fx"];
        float fy = fSetting["Camera2.fy"];
        float cx = fSetting["Camera2.cx"];
        float cy = fSetting["Camera2.cy"];
        vCalibrationRight = {fx, fy, cx, cy};
        float k1 = fSetting["Camera2.k1"];
        float k2 = fSetting["Camera2.k2"];
        float p1 = fSetting["Camera2.p1"];
        float p2 = fSetting["Camera2.p2"];
        vPinHoleDistorsionRight = {k1, k2, p1, p2};
    }
    // Tlr
    cv::Mat Tlr_mat = fSetting["T_c1_c2"].mat();
//    LOG(INFO) << Tlr_mat;
    Sophus::SE3<float> Tlr;
    {
        Eigen::Matrix<double, 3, 3> eigMat;
        eigMat << Tlr_mat.at<float>(0, 0), Tlr_mat.at<float>(0, 1), Tlr_mat.at<float>(0, 2),
                Tlr_mat.at<float>(1, 0), Tlr_mat.at<float>(1, 1), Tlr_mat.at<float>(1, 2),
                Tlr_mat.at<float>(2, 0), Tlr_mat.at<float>(2, 1), Tlr_mat.at<float>(2, 2);
        Eigen::Quaternionf q(eigMat.cast<float>());
        Eigen::Matrix<float, 3, 1> t;
        cv::Mat cvVector = Tlr_mat.rowRange(0, 3).col(3);
        t << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);
        Tlr = SE3f{q, t};
    }

    string base_dir = "/home/zhonglingjun/GXT/MySLAM/src/my_slam/datas";
    string left_path = base_dir + "/008_left.png";
    string right_path = base_dir + "/008_right.png";

    Mat left_image, right_image;
    left_image = imread(left_path, cv::IMREAD_GRAYSCALE);
    right_image = imread(right_path, cv::IMREAD_GRAYSCALE);
#ifdef SHOW_RAW
    imshow("left_raw", left_image);
    imshow("right_raw", right_image);
#endif
    int nfeatures = 1000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 15, minThFAST = 7;
    ORBextractor extractorLeft{nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST};
    ORBextractor extractorRight{nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST};

    /// 提取特征
    vector<KeyPoint> mvKeys, mvKeysRight;
    cv::Mat mDescriptors, mDescriptorsRight;
    vector<int> vLapping = {0, 0};
    int monoLeft = extractorLeft(left_image, cv::Mat(), mvKeys, mDescriptors, vLapping);
    int monoRight = extractorRight(right_image, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);

    // 左目特征点去畸变

    // 计算双目匹配



    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;
}