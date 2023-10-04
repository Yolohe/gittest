
#include "iostream"
#include "sstream"
#include "fstream"
#include "algorithm"
#include "cstring"
using namespace std;

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "../inc/camera.h"
using namespace cv;

#include "librealsense2/rs.hpp"
#include "librealsense2/rsutil.h"

// 获取深度像素对应长度单位转换
float get_depth_scale(const rs2::device &dev)
{
  // 前往摄像头传感器
  for (rs2::sensor &sensor : dev.query_sensors()) // 使用与，两者发生一个既可
  {
    // 检查是否有深度图像
    if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) // 检查是否有深度图
    {
      return dpt.get_depth_scale(); // 在数组中返回数值
    }
  }
  throw std::runtime_error("Device Error!"); // 发生错误打印
}

// 深度图对齐到彩色图像
Mat align_Depth2Color(Mat depth, const Mat &color, rs2::pipeline_profile profile)
{
  // 定义数据流深度与图像//auto默认类型rs2::video_stream_profile
  auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>(); // 使用auto可以直接定义之前的数据类型
  auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

  // 获得内部参数(使用const只能在内部使用，与静态变量相似)
  const auto intrinDepth = depth_stream.get_intrinsics(); // 只能在函数内使用
  const auto intrinColor = color_stream.get_intrinsics();

  // 直接获取从深度相机坐标系到彩色摄像头坐标系的欧拉转换矩阵
  rs2_extrinsics extrinDepth2Color; // 声明
  rs2_error *error;
  rs2_get_extrinsics(depth_stream, color_stream, &extrinDepth2Color, &error);

  // 平面点定义
  float pd_uv[2], pc_uv[2]; // 定义数组
  // 空间点定义
  float Pdc3[3], Pcc3[3];

  // 获得深度像素与现实单位比例
  float depth_scale = get_depth_scale(profile.get_device());
  int y, x;
  // 初始化结果
  Mat result = Mat(color.rows, color.cols, CV_16U, Scalar(0));
  // 对深度图像处理
  for (int row = 0; row < depth.rows; row++)
  {
    for (int col = 0; col < depth.cols; col++)
    {
      pd_uv[0] = col;
      pd_uv[1] = row;
      // 得到当前的深度数值
      uint16_t depth_value = depth.at<uint16_t>(row, col);
      // 换算单位
      float depth_m = depth_value * depth_scale; // 换算成米
      // 深度图像的像素点转换为坐标下三维点
      rs2_deproject_pixel_to_point(Pdc3, &intrinDepth, pd_uv, depth_m);
      // 深度相机坐标系的三维点转化到彩色的坐标系下
      rs2_transform_point_to_point(Pcc3, &extrinDepth2Color, Pdc3);
      // 彩色摄像头坐标系下深度三维点映射到二位平面上
      rs2_project_point_to_pixel(pc_uv, &intrinColor, Pcc3);

      // 取得映射后的（u,v）
      x = (int)pc_uv[0]; // 处理后的数据
      y = (int)pc_uv[1];

      x = x < 0 ? 0 : x;
      x = x > depth.cols - 1 ? depth.cols - 1 : x;
      y = y < 0 ? 0 : y;
      y = y > depth.rows - 1 ? depth.rows - 1 : y;

      result.at<uint16_t>(y, x) = depth_value;
    }
  }
  return result; // 返回与彩色图对齐的图像
}

void measure_distance(Mat &color, Mat depth, cv::Size range, rs2::pipeline_profile profile) // 声明profile
{
  // 获得深度像素与现实单位比例
  float depth_scale = get_depth_scale(profile.get_device());
  // 定义图像中心点
  cv::Point center(color.cols / 2, color.rows / 2);
  // 定义计算距离的范围
  cv::Rect RectRange(center.x - range.width / 2, center.y - range.height / 2, range.width, range.height);
  // 画出范围
  float distance_sum = 0;
  int effective_pixel = 0;
  for (int y = RectRange.y; y < RectRange.y + RectRange.height; y++)
  {
    for (int x = RectRange.x; x < RectRange.x + RectRange.width; x++)
    {
      // 不是0就有位置信息
      if (depth.at<uint16_t>(y, x)) // 出现位置信息
      {
        distance_sum += depth_scale * depth.at<uint16_t>(y, x);
        effective_pixel++;
      }
    }
  }
  cout << "有效像素点：" << effective_pixel << endl; // 输出数据
  float effective_distance = distance_sum / effective_pixel;
  cout << "目标距离：" << effective_distance << "m" << endl;
  char distance_str[30];
  sprintf(distance_str, "The distance is:%f m", effective_distance);
  cv::rectangle(color, RectRange, Scalar(0, 0, 255), 2, 8);
  cv::putText(color, (string)distance_str, cv::Point(color.cols * 0.02, color.rows * 0.05),
              cv::FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2, 8);
}

int main()
{
  const char *depth_win = "depth_Image";
  namedWindow(depth_win, WINDOW_AUTOSIZE); // 开启窗口
  const char *color_win = "color_Image";
  namedWindow(color_win, WINDOW_AUTOSIZE);

  // 深度图像颜色map
  rs2::colorizer c; // 声明

  // 创建数据管道
  rs2::pipeline pipe;
  rs2::config pipe_config;
  pipe_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
  pipe_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

  // 开始函数返回值的profile
  rs2::pipeline_profile profile = pipe.start(pipe_config);

  // 定义一个变量从深度转化到距离
  float depth_clipping_distance = 1.f;

  // 声明数据
  auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
  auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

  // 获得内参
  auto intrinDepth = depth_stream.get_intrinsics();
  auto intrinColor = color_stream.get_intrinsics();

  // 直接获取从深度摄像头到彩色摄像的欧式变化矩阵
  auto extrinDepth2Color = depth_stream.get_extrinsics_to(color_stream);

  while (cvGetWindowHandle(depth_win) && cvGetWindowHandle(color_win))
  {
    // 堵塞程序到新的帧出现
    rs2::frameset frameset = pipe.wait_for_frames();
    // 取得深度图和彩色图
    rs2::frame color_frame = frameset.get_color_frame();                        // 取得彩色图像
    rs2::frame depth_frame = frameset.get_depth_frame();                        // 取得深度图像
    rs2::frame depth_frame_4_show = frameset.get_depth_frame().apply_filter(c); // c为rs2::colorizer
    // 获得宽高
    const int depth_w = depth_frame.as<rs2::video_frame>().get_width();
    const int depth_h = depth_frame.as<rs2::video_frame>().get_height();
    const int color_w = color_frame.as<rs2::video_frame>().get_width();
    const int color_h = color_frame.as<rs2::video_frame>().get_height();

    // 创造opencv类型传输数据
    Mat depth_image(Size(depth_w, depth_h),
                    CV_16U, (void *)depth_frame.get_data(), Mat::AUTO_STEP);
    Mat depth_image_4_show(Size(depth_w, depth_h),
                           CV_8UC3, (void *)depth_frame_4_show.get_data(), Mat::AUTO_STEP);
    Mat color_image(Size(color_w, color_h),
                    CV_8UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);

    // 实现深度图对齐彩色图
    Mat result = align_Depth2Color(depth_image, color_image, profile); // 调用对齐函数
    measure_distance(color_image, result, cv::Size(20, 20), profile);

    // 显示
    imshow(depth_win, depth_image_4_show); // 在深度图显示深度图像
    imshow(color_win, color_image);        // 在彩色图显示彩色

    waitKey(1); // 延时1
  }
  return 0;
}
