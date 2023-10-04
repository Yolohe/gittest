#include <librealsense2/rs.hpp>
#include "../inc/OpencvHead.h"
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char **argv)
{
 
    // 初始化相机和数据流
    cout << "Line 6 - c 的值是 " << endl ;
    rs2::pipeline pipeline;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipeline.start(cfg);
 
    // 初始化窗口
    cv::namedWindow("RGB Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);
 
    while (true) {
        // 等待获取帧数据
        rs2::frameset frames = pipeline.wait_for_frames();
 
        // 获取RGB图像
        rs2::video_frame color = frames.get_color_frame();
        cv::Mat rgb_image(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
 
        // 获取深度图像
        rs2::depth_frame depth = frames.get_depth_frame();
        cv::Mat depth_image(cv::Size(depth.get_width(), depth.get_height()), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
 
        // 归一化深度图像以便显示
        double min, max;
        cv::minMaxIdx(depth_image, &min, &max);
        cv::Mat normalized_depth_image;
        cv::convertScaleAbs(depth_image, normalized_depth_image, 255.0 / max);
 
        // 将RGB图像和深度图像显示出来
        cv::imshow("RGB Image", rgb_image);
        cv::imshow("Depth Image", normalized_depth_image);
 
        // 等待按键事件
        if (cv::waitKey(1) >= 0)
            break;
    }
 
    return 0;
}