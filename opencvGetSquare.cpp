/**
 * @file getSquare.cpp
 * @author mylizi
 * @brief 调参：cannyMaxThresh cannyMinThresh approxPolyDP第三个参数 判断面积 滤波函数和参数
 * 效果：边缘提取参数调好的话提取效果挺好的，但就是没法稳定提取矩形，问题在轮廓提取和折线化。折线化问题是最大的
 * 相机稳定不动时都无法稳定地提取矩形，抗扰动性很差。
 * 颜色区域提取因为黄色和橙色很相近，只要障碍物的黄色杂点橙就很难办。而且规则上写的地板是深黄色，示意图用的又是橙色，不能保证效果。
 * 基尔霍夫直线变换效果不佳。参数复用性不大。
 * @version 0.2
 * @date 2022-12-11
 */
#include "../inc/opencvGetSquare.h"

using namespace std;
using namespace cv;

int cannyMaxThresh = 200, cannyMinThresh = 100, judgeArea = 500, medianBlurParam = 7, N = 5;
float approxPolyDPThirdParam = 0.025; // 折线化第三个参数

const char *wndname = "Square Detection Demo";

// 求两向量的余弦值
static double angle(Point pt1, Point pt2, Point pt0)
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1 * dx2 + dy1 * dy2) /
         sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10); // -10次方防止除数为0
}

/**
 * @brief 提取指定颜色，输入BGR图像
 * @param Hmin 颜色最大值，0-179
 * @param Hmax 颜色最大值，0-179
 * @param Smin 饱和度最小值，0-255
 * @param Smax 饱和度最大值，0-255
 */
void detectColor(const Mat &srcImg, Mat &dstImg, int Hmin, int Hmax, int Smin, int Smax, int Vmin, int Vmax)
{
  Mat hsv;
  cvtColor(srcImg, hsv, COLOR_BGR2HSV);
  vector<Mat> channels;
  split(hsv, channels);

  Mat Hmask1, Hmask2, HMask;
  threshold(channels[0], Hmask1, Hmax, 255, THRESH_BINARY_INV);
  threshold(channels[0], Hmask2, Hmin, 255, THRESH_BINARY);
  if (Hmin < Hmax)
  {
    HMask = Hmask1 & Hmask2; // 提取区域中的颜色
  }
  else
  {
    HMask = Hmask1 | Hmask2; // 除去区域中的颜色
  }
  Mat SMask;
  inRange(channels[1], Smin, Smax, SMask);
  Mat VMask;
  inRange(channels[2], Vmin, Vmax, VMask);

  // 测试掩膜效果
  imshow(wndname, HMask);
  waitKey(0);
  imshow(wndname, SMask);
  waitKey(0);
  imshow(wndname, VMask);
  waitKey(0);

  Mat mask;
  mask = HMask & SMask & VMask;
  srcImg.copyTo(dstImg, mask);
}

// 寻找长方形(不提取颜色直接处理)
void findSquares(const Mat &image, vector<vector<Point>> &squares)
{
  squares.clear();

  Mat etimg = image.clone();
  Mat timg;
  medianBlur(image, timg, medianBlurParam); // 中值滤波
  // int d = 50;
  // bilateralFilter(etimg, timg, d, d * 2, d / 2); // 边缘滤波

  // 测试滤波效果
  // imshow(wndname, timg);
  // waitKey(0);

  Mat grayImg(timg.size(), CV_8U), edgeImg;

  vector<vector<Point>> contours; // 相当于二维数组，每个元素类型为向量(x,y)

  // 在每个通道中寻找长方形
  for (int c = 0; c < 3; c++)
  {
    int ch[] = {c, 0};
    mixChannels(&timg, 1, &grayImg, 1, ch, 1); // 把timg中的c通道复制到grayImg的0通道中

    for (int l = 0; l < N; l++)
    {
      if (l == 0)
      {
        Canny(grayImg, edgeImg, cannyMinThresh, cannyMaxThresh, 5); // canny边缘提取
        // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));// 自定义核
        dilate(edgeImg, edgeImg, Mat(),
               Point(-1, -1)); // 膨胀,Mat()为3x3的矩形核元素，point(-1,-1)表明锚点在元素的中心位置

        // // 测试边缘提取效果
        // imshow(wndname, edgeImg);
        // waitKey(0);
      }
      else
      {
        // tgray(x,y) = gray(x,y) >= (l+1)*255/N ? 0 : 255
        edgeImg = grayImg >= (l + 1) * 255 / N;
      }

      findContours(edgeImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); // 从边缘图中提取轮廓

      vector<Point> approx;

      // // 测试采样效果
      // Mat testImg(timg.size(), CV_8UC3, Scalar(0, 0, 0));

      // 检查每一组的点是否能组成矩形
      for (size_t i = 0; i < contours.size(); i++)
      {
        // 对连续光滑曲线采样，进行折线化，以指定精度近似多边形曲线，输出为多边形点集
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * approxPolyDPThirdParam,
                     true); // 精度和周长成比例。arcLength计算轮廓周长，第三个参数为原始曲线和近似曲线的最大距离

        // // 测试采样效果
        // for (size_t i = 0; i < approx.size(); i++)
        // {
        //     circle(testImg, approx[i], 1, Scalar(255, 255, 255), -1);
        // }

        // 四个顶点并且面积大于一定范围并且是凸图形
        if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > judgeArea && isContourConvex(Mat(approx)))
        {
          double maxCosine = 0;
          double cosine = 0;
          // 找四个角余弦值的最大值（即角度最小）
          for (int j = 1; j < 5; j++)
          {
            if (j == 1)
              cosine = fabs(angle(approx[j % 4], approx[j + 2], approx[j - 1]));
            else
              cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
            maxCosine = MAX(maxCosine, cosine);
          }

          // 大于78.463度则认为是矩形
          if (maxCosine < 0.2)
            squares.push_back(approx);
        }
      }
      // // 测试采样效果
      // imshow(wndname, testImg);
      // waitKey(0);
    }
  }
}

// 寻找长方形（提取颜色后再处理）
void findSquaresAfterDetectColor(const Mat &srcImg, vector<vector<Point>> &squares)
{
  squares.clear();
  Mat grayImg, edgeImg;
  cvtColor(srcImg, grayImg, COLOR_BGR2GRAY); // 转灰度处理

  vector<vector<Point>> contours; // 相当于二维数组，每个元素类型为向量(x,y)

  Canny(grayImg, edgeImg, cannyMinThresh, cannyMaxThresh, 5); // canny边缘提取
  // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));// 自定义核
  dilate(edgeImg, edgeImg, Mat(), Point(-1, -1)); // 膨胀,Mat()为3x3的矩形核元素，point(-1,-1)表明锚点在元素的中心位置

  // 测试边缘提取效果
  imshow(wndname, edgeImg);
  waitKey(0);

  findContours(edgeImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); // 从边缘图中提取轮廓

  vector<Point> approx;

  // 测试采样效果
  Mat testImg(grayImg.size(), CV_8UC3, Scalar(0, 0, 0));

  // 检查每一组的点是否能组成矩形
  for (size_t i = 0; i < contours.size(); i++)
  {
    // 对连续光滑曲线采样，进行折线化，以指定精度近似多边形曲线，输出为多边形点集
    approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * approxPolyDPThirdParam,
                 true); // 精度和周长成比例。arcLength计算轮廓周长，第三个参数为原始曲线和近似曲线的最大距离

    // 测试采样效果
    for (size_t i = 0; i < approx.size(); i++)
    {
      circle(testImg, approx[i], 1, Scalar(255, 255, 255), -1);
    }

    // 四个顶点并且面积大于一定范围并且是凸图形
    if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > judgeArea && isContourConvex(Mat(approx)))
    {
      double maxCosine = 0;
      double cosine = 0;
      // 找四个角余弦值的最大值（即角度最小）
      for (int j = 1; j < 5; j++)
      {
        if (j == 1)
          cosine = fabs(angle(approx[j % 4], approx[j + 2], approx[j - 1]));
        else
          cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
        maxCosine = MAX(maxCosine, cosine);
      }

      // 根据实际来定吧
      if (maxCosine < 0.6)
        squares.push_back(approx);
    }
  }
  // 测试采样效果
  imshow(wndname, testImg);
  waitKey(0);
}

// 利用霍夫变换进行直线检测
void findlinesUsingHough(const Mat &srcImg)
{
  Mat grayImg, edgeImg;
  cvtColor(srcImg, grayImg, COLOR_BGR2GRAY); // 转灰度处理

  Canny(grayImg, edgeImg, cannyMinThresh, cannyMaxThresh, 5); // canny边缘提取
  // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));// 自定义核
  dilate(edgeImg, edgeImg, Mat(), Point(-1, -1)); // 膨胀,Mat()为3x3的矩形核元素，point(-1,-1)表明锚点在元素的中心位置

  // 测试边缘提取效果
  imshow(wndname, edgeImg);
  waitKey(0);

  vector<Vec4f> plines;
  HoughLinesP(edgeImg, plines, 1, CV_PI / 180.0, 5, 100, 3);
  Mat image;
  srcImg.copyTo(image);
  drawLines(image, plines);
}

// 把长方形画在图上
void drawSquares(Mat &image, const vector<vector<Point>> &squares)
{
  for (size_t i = 0; i < squares.size(); i++)
  {
    const Point *p = &squares[i][0];

    int n = (int)squares[i].size();
    // 太靠近图像边缘的舍弃掉
    if (p->x > 3 && p->y > 3)
      polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
  }

  // 测试
  imshow(wndname, image);
}

// 把直线画在图上
void drawLines(Mat &image, const vector<Vec4f> &plines)
{
  Scalar color = Scalar(0, 255, 0);
  for (size_t i = 0; i < plines.size(); i++)
  {
    Vec4f hline = plines[i];
    line(image, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
  }
  imshow(wndname, image);
}

// 提取颜色的测试
void test(Mat &srcimg)
{
  cv::Mat hsvImage;
  colorHist cHis;
  std::vector<colorHist> features;
  cv::cvtColor(srcimg, hsvImage, cv::COLOR_BGR2HSV);
  int h_bins = 30;
  int s_bins = 50;
  int histSize[] = {h_bins, s_bins}; // 直方图分成的区间个数
  float h_ranges[] = {0, 180};
  float s_ranges[] = {0, 256};
  const float *ranges[] = {h_ranges, s_ranges};
  int channels[] = {0, 1};
  cv::Mat color_features;
  calcHist(&hsvImage, 1, channels, cv::Mat(), color_features, 2, histSize, ranges);
  normalize(color_features, color_features, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  for (int i = 0; i < color_features.rows; i++) // 将直方图均衡化后的结果转为另一种形式存储在features里
  {
    for (int j = 0; j < color_features.cols; j++)
    {
      cHis.feat = color_features.at<float>(i, j);
      cHis.x = i;
      cHis.y = j;
      features.push_back(cHis);
    }
  }
  cout << color_features.rows << " " << color_features.cols << endl;
}

// // 测试
// int main()
// {
//     char path[] = "/media/jojo/Data/cpp/camera/testImg/盒3一面.jpg";
//     Mat srcimg = imread(path);
//     resize(srcimg, srcimg, Size(1000, 800));

//     test(srcimg);

//     // medianBlur(srcimg, srcimg, medianBlurParam);

//     // 基尔霍夫直线变换
//     // vector<vector<Point>> squares;
//     // findlinesUsingHough(srcimg);

//     // 颜色区域提取
//     // Mat dst;
//     // detectColor(srcimg, dst, 20, 34, 43, 255, 46, 255);
//     // imshow(wndname, dst);
//     // waitKey(0);

//     // 找矩形
//     // vector<vector<Point>> squares;
//     // findSquaresAfterDetectColor(dst, squares);
//     // drawSquares(srcimg, squares);

//     // namedWindow(wndname, WINDOW_GUI_NORMAL);
//     // imshow(wndname, dst);
//     waitKey(0);
// }
