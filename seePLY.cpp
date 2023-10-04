#include "../inc/PCLHead.h"
#include <pcl/surface/mls.h>
#include <iostream>

using namespace std;
using namespace pcl;
typedef PointXYZ PointT;
int main(void)
{
    PointCloud<PointXYZ>::Ptr sphere(new PointCloud<PointXYZ>), source(new PointCloud<PointXYZ>);

    visualization::PCLVisualizer viewer("viewer");

    int v1(0); // 创建视口v1
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    io::loadPLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/SPHERE0.ply", *sphere);
    viewer.addPointCloud(sphere, "sphere");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 0.5, 0, 0.5, "sphere", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sphere", v1);
    viewer.addCoordinateSystem(50);
    // io::loadPLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/source_downsampled.ply", *source);
    MovingLeastSquares<PointXYZ, PointNormal> mls;
    mls.setComputeNormals(true);
    mls.setPolynomialOrder(true);
    // io::savePLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere6*.ply", *sphere);

    PointT center0;
    center0.x = 0;
    center0.y = 0;
    center0.z = 0;
    PointT center1;
    center1.x = 0;
    center1.y = 0;
    center1.z = 0;
    PointT center2;
    center2.x = 0;
    center2.y = 0;
    center2.z = 0;
    PointT center3;
    center3.x = 0;
    center3.y = 0;
    center3.z = 0;
    PointT center4;
    center4.x = 0;
    center4.y = 0;
    center4.z = 0;
    PointT center5;
    center5.x = 0;
    center5.y = 0;
    center5.z = 0;
    PointT center6;
    center6.x = 0;
    center6.y = 0;
    center6.z = 0;
    PointT center7;
    center7.x = 0;
    center7.y = 0;
    center7.z = 0;
    PointT center8;
    center8.x = 0;
    center8.y = 0;
    center8.z = 0;
    PointT center9;
    center9.x = 0;
    center9.y = 0;
    center9.z = 0;
    PointT center10;
    center10.x = 0;
    center10.y = 0;
    center10.z = 0;
    for (auto c : sphere->points)
    {
        center0.x += c.x;
        center0.y += c.y;
        center0.z += c.z;
    }
    center0.x /= sphere->points.size();
    center0.y /= sphere->points.size();
    center0.z /= sphere->points.size();

    for (auto c : sphere->points)
    {
        center1.x += c.x * c.x;
        center1.y += c.y * c.y;
        center1.z += c.z * c.z;
    }
    center1.x /= sphere->points.size();
    center1.y /= sphere->points.size();
    center1.z /= sphere->points.size();

    for (auto c : sphere->points)
    {
        center2.x += c.x * c.y;
        center2.y += c.x * c.z;
        center2.z += c.y * c.z;
    }
    center2.x /= sphere->points.size();
    center2.y /= sphere->points.size();
    center2.z /= sphere->points.size();
    for (auto c : sphere->points)
    {
        center3.x += c.x * c.x * c.x;
        center3.y += c.x * c.x * c.y;
        center3.z += c.x * c.x * c.z;
    }
    center3.x /= sphere->points.size();
    center3.y /= sphere->points.size();
    center3.z /= sphere->points.size();
    for (auto c : sphere->points)
    {
        center4.x += c.x * c.y * c.y;
        center4.y += c.x * c.z * c.z;
        center4.z += c.y * c.y * c.y;
    }
    center4.x /= sphere->points.size();
    center4.y /= sphere->points.size();
    center4.z /= sphere->points.size();
    for (auto c : sphere->points)
    {
        center5.x += c.y * c.y * c.z;
        center5.y += c.y * c.z * c.z;
        center5.z += c.z * c.z * c.z;
    }
    center5.x /= sphere->points.size();
    center5.y /= sphere->points.size();
    center5.z /= sphere->points.size();
    center6.x = center1.x - center0.x * center0.x;
    center6.y = center2.x - center0.x * center0.y;
    center6.z = center2.y - center0.x * center0.z;
    center7.x = center2.x - center0.x * center0.y;
    center7.y = center1.y - center0.y * center0.y;
    center7.z = center2.z - center0.y * center0.z;
    center8.x = center2.y - center0.x * center0.z;
    center8.y = center2.z - center0.y * center0.z;
    center8.z = center1.z - center0.z * center0.z;
    center9.x = center3.x - center0.x * center1.x + center4.x - center0.x * center1.y + center4.y - center0.x * center1.z;
    center9.y = center3.y - center0.y * center1.x + center4.z - center0.y * center1.y + center5.y - center0.y * center1.z;
    center9.z = center3.z - center0.z * center1.x + center5.x - center0.z * center1.y + center5.z - center0.z * center1.z;
    center10.x = center9.x / 2;
    center10.y = center9.y / 2;
    center10.z = center9.z / 2;

    Eigen::Matrix3f B;
    B << center6.x, center6.y, center6.z,
        center7.x, center7.y, center7.z,
        center8.x, center8.y, center8.z;
    Eigen::Vector3f b(center10.x, center10.y, center10.z);
    Eigen::Vector3f C = B.colPivHouseholderQr().solve(b);
    std::cout << C << std::endl;
    PointT center;
    center.x = C.x();
    center.y = C.y();
    center.z = C.z();
    int v2(0); // 创建视口v2
    viewer.addSphere(center, 5, 0, 0, 1, "center", 0);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    io::loadPLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/source.ply", *source);
    viewer.addPointCloud(source, "source");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "source", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "source", v2);
    viewer.addCoordinateSystem(50);
    while (!viewer.wasStopped())
    {
        viewer.spin();
    }

    return 0;
}
// #include "../inc/pointcloud.h"
// int main() // 随机填充
// {
//     // 初始化点云
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     // 设置点云大小
//     cloud->points.resize(500);
//     cloud->width = 100;
//     cloud->height = 100;
//     cloud->points.resize(cloud->width * cloud->height);

//     // 填充点云
//     for (size_t i = 0; i < cloud->points.size(); ++i)
//     {
//         // cloud->points[i].x = 102.4 * rand() / (RAND_MAX + 0.01f);
//         // cloud->points[i].y = 102.4 * rand() / (RAND_MAX + 0.01f);
//         // cloud->points[i].z = 102.4 * rand() / (RAND_MAX + 0.01f);
//         cloud->points[i].x = 80;
//         cloud->points[i].y = 80;
//         cloud->points[i].z = 80;
//     }
//     pcl::io::savePLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/source3.ply", *cloud);
//     // 声明视窗
//     pcl::visualization::PCLVisualizer viewer("viewer");
//     // 设置视窗背景色

//     // 预处理点云颜色
//     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> magenta(cloud, 255, 0, 255);
//     // 把点云加载到视窗
//     viewer.addPointCloud(cloud, magenta, "cloud");
//     // 设置点云大小
//     viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
//     viewer.addCoordinateSystem(50);
//     viewer.spin();
// }

// #include <pcl/io/ply_io.h>
// #include <pcl/sample_consensus/ransac.h>
// #include <pcl/sample_consensus/sac_model_sphere.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include "../inc/PCLHead.h"
// using namespace std;

// int main()
// {
//     //-------------------------- 加载点云 --------------------------
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     if (pcl::io::loadPLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/source2.ply", *cloud) < 0)
//     {
//         PCL_ERROR("点云读取失败！\n");
//         return -1;
//     }
//     cout << "加载点云点数：" << cloud->points.size() << endl;
//     //----------------------------拟合球----------------------------
//     pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_sphere(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));
//     pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_sphere);
//     ransac.setDistanceThreshold(0.1); // 设置距离阈值，与球面距离小于0.1的点作为内点
//     ransac.computeModel();            // 执行模型估计

//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sphere(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::ExtractIndices<pcl::PointXYZ> extract;
//     //---------------------------提取内点---------------------------
//     vector<int> inliers;        // 存储内点索引的容器
//     ransac.getInliers(inliers); // 提取内点索引

//     pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *cloud_sphere);
//     //-------------------------输出球模型参数-----------------------
//     Eigen::VectorXf coefficient;
//     ransac.getModelCoefficients(coefficient);
//     cout << "球面方程为：\n"
//          << "(x - " << coefficient[0] << ") ^ 2 + (y - " << coefficient[1]
//          << ") ^ 2 + (z - " << coefficient[2] << ")^2 = " << coefficient[3] << " ^2" << endl;
//     extract.setNegative(false);
//     extract.filter(*cloud_sphere);
//     pcl::io::savePLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere**.ply", *cloud_sphere);
//     //----------------------------结果可视化--------------------------
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("拟合结果"));

//     // viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
//     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud");
//     // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

//     viewer->addPointCloud<pcl::PointXYZ>(cloud_sphere, "sphere");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "sphere");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sphere");
//     while (!viewer->wasStopped())
//     {
//         viewer->spin();
//     }

//     return 0;
// }