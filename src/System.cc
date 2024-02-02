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


int DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

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

    auto mnScaleLevels = extractorLeft.GetLevels();
    auto mfScaleFactor = extractorLeft.GetScaleFactor();
    auto mfLogScaleFactor = log(mfScaleFactor);
    auto mvScaleFactors = extractorLeft.GetScaleFactors();
    auto mvInvScaleFactors = extractorLeft.GetInverseScaleFactors();
    auto mvLevelSigma2 = extractorLeft.GetScaleSigmaSquares();
    auto mvInvLevelSigma2 = extractorLeft.GetInverseScaleSigmaSquares();

    /// 提取特征
    vector<KeyPoint> mvKeys, mvKeysRight, mvKeysUn;
    cv::Mat mDescriptors, mDescriptorsRight;
    vector<int> vLapping = {0, 0};
    int monoLeft = extractorLeft(left_image, cv::Mat(), mvKeys, mDescriptors, vLapping);
    int monoRight = extractorRight(right_image, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);

    // 左目特征点去畸变 UndistortKeyPoints()

    Mat mK = Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = vCalibrationLeft[0];
    mK.at<float>(1, 1) = vCalibrationLeft[1];
    mK.at<float>(0, 2) = vCalibrationLeft[2];
    mK.at<float>(1, 2) = vCalibrationLeft[3];

    Mat mDistCoef = Mat::zeros(4, 1, CV_32F);
    mDistCoef.at<float>(0) = vPinHoleDistorsionLeft[0];
    mDistCoef.at<float>(1) = vPinHoleDistorsionLeft[1];
    mDistCoef.at<float>(2) = vPinHoleDistorsionLeft[2];
    mDistCoef.at<float>(3) = vPinHoleDistorsionLeft[3];

    int N = mvKeys.size();
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }
    cv::Mat K = (cv::Mat_<float>(3, 3)
            << vCalibrationLeft[0], 0.f, vCalibrationLeft[2], 0.f, vCalibrationLeft[1], vCalibrationLeft[3], 0.f, 0.f, 1.f);
    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++) {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }

    // 计算双目匹配 ComputeStereoMatches()
    /* 两帧图像稀疏立体匹配（即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
     * 输入：两帧立体矫正后的图像img_left 和 img_right 对应的orb特征点集
     * 过程：
          1. 行特征点统计. 统计img_right每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
          2. 粗匹配. 根据步骤1的结果，对img_left第i行的orb特征点pi，在img_right的第i行上的orb特征点集中搜索相似orb特征点, 得到qi
          3. 精确匹配SAD. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
          4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取float精度的真实视差
          5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点。
          6. 删除离群点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
     * 输出：稀疏特征点视差图/深度图（亚像素精度）mvDepth 匹配结果 mvuRight
     */
    vector<float> mvuRight = vector<float>(N, -1.0f);
    vector<float> mvDepth = vector<float>(N, -1.0f);

    const int thOrbDist = (100 + 50) / 2; // ORBmatcher::TH_HIGH(100)+ORBmatcher::TH_LOW(50)
    const int nRows = extractorLeft.mvImagePyramid[0].rows;

    // 行统计右图特征点
    vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());
    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(200);
    const int Nr = mvKeysRight.size();

    // Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for (int iR = 0; iR < Nr; iR++) {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++)
            vRowIndices[yi].push_back(iR);
    }
    // Step 2 -> 3. 粗匹配 + 精匹配
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[u - fx, u], 而不需要遍历每一行所有的像素

    // mb = Tlr_.translation().norm(); // baseline
    // mbf = mb * fx
    // 视差公式, z=(fb)/d, d= u_l - u_r
    // z(深度), d(视差), f(焦距), b(基线)
    float mb = Tlr.translation().norm();
    float mbf = mb * vCalibrationLeft[0];

    // 右图行搜索限制 [u- fx, u]
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf / minZ;

    // 保存sad块匹配相似度和左图特征点索引，精确匹配使用
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);
    for (int iL = 0; iL < N; iL++) {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave; // 金字塔的层次
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;
        // 获取左图特征点pi所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];
        if (vCandidates.empty())
            continue;
        const float minU = uL - maxD;// ul - fx
        const float maxU = uL - minD;// ul - 0
        if (maxU < 0)
            continue;
        int bestDist = 100;
        size_t bestIdxR = 0;
        const cv::Mat &dL = mDescriptors.row(iL);

        // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
        for (unsigned long iR: vCandidates) {
            const cv::KeyPoint &kpR = mvKeysRight[iR];
            // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;
            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;
            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if (uR >= minU && uR <= maxU) {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = DescriptorDistance(dL, dR);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        // Step 3. 精确SAD匹配.
        if (bestDist < thOrbDist) {
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            // 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x * scaleFactor);
            const float scaledvL = round(kpL.pt.y * scaleFactor);
            const float scaleduR0 = round(uR0 * scaleFactor);

            // 滑动窗口搜索, 类似模版卷积或滤波
            // w表示sad相似度的窗口半径
            const int w = 5;
            // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
            cv::Mat IL = extractorLeft.mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(
                    scaleduL - w, scaleduL + w + 1);

            int bestDist = INT_MAX; //初始化最佳相似度
            int bestincR = 0;   // 通过滑动窗口搜索优化，得到的列坐标偏移量
            const int L = 5;    //滑动窗口的滑动范围为（-L, L）
            // 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2 * L + 1);
            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 - 最大窗口滑动范围 - 图像块尺寸
            // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;
            // 判断搜索是否越界
            if (iniu < 0 || endu >= extractorRight.mvImagePyramid[kpL.octave].cols)
                continue;
            // 在搜索范围内从左到右滑动，并计算图像块相似度
            for (int incR = -L; incR <= +L; incR++) {
                cv::Mat IR = extractorRight.mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                scaledvL + w + 1).colRange(
                        scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                // ORB3 没有 图像块均值归一化，降低亮度变化对相似度计算的影响
                // IR.convertTo(IR, CV_32F);
                // IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                // sad 计算
                float dist = cv::norm(IL, IR, cv::NORM_L1);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestincR = incR;
                }

                vDists[L + incR] = dist;
            }

            if (bestincR == -L || bestincR == L)
                continue;

            // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优视差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              .
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];

            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));
            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if (deltaR < -1 || deltaR > 1)
                continue;

            // 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave] * ((float) scaleduR0 + (float) bestincR + deltaR);

            float disparity = (uL - bestuR);
            // 计算深度
            if (disparity >= minD && disparity < maxD) {
                if (disparity <= 0) {
                    disparity = 0.01;
                    bestuR = uL - 0.01;
                }
                mvDepth[iL] = mbf / disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int, int>(bestDist, iL));
            }
        }
    }
    // Step 6. 删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0; i--) {
        if (vDistIdx[i].first < thDist)
            break;
        else {
            mvuRight[vDistIdx[i].second] = -1;
            mvDepth[vDistIdx[i].second] = -1;
        }
    }

    for (size_t i = 0; i < mvDepth.size(); i++) {
        float z = mvDepth[i];
        if (z > 0) {
            float u = mvKeysUn[i].pt.x;
            float v = mvKeysUn[i].pt.y;
            float x = (u - vCalibrationLeft[0]) * z * (1. / vCalibrationLeft[2]);
            float y = (v - vCalibrationLeft[1]) * z * (1. / vCalibrationLeft[3]);
            Eigen::Vector3f x3Dc(x, y, z);
//            x3D = mRwc * x3Dc + mOw;
            LOG(INFO) << x3Dc.transpose();
        }
    }
    // Draw imshow
    cv::Mat left_image_draw, right_image_draw;
    drawKeypoints(left_image, mvKeys, left_image_draw);
    drawKeypoints(right_image, mvKeysRight, right_image_draw);
//    cv::Mat undistortedImage;
//    cv::undistort(left_image,undistortedImage, K, mDistCoef);
//    drawKeypoints(undistortedImage, mvKeysUn, undistortedImage);
//    imshow("left undistorted", undistortedImage);
    imshow("left_raw_draw", left_image_draw);
//    imshow("right_raw_draw", right_image_draw);

    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;
}