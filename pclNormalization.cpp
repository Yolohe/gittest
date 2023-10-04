#include "../inc/pclNormalization.h"
using namespace std;
struct Point
{
    double x, y, z;
};
vector<Point> findMaxAndMin(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);
pcl::PointCloud<pcl::PointXYZ>::Ptr normialize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, vector<Point> MaxAndMin, string filename);
int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere**.ply", *cloud) == -1) //*打开点云文件
    {
        PCL_ERROR("Couldn't read ply\n");
        return (-1);
    }

    // cout << "Load "
    //      << "/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere10.ply" << endl;
    //    normialize(cloud,findMaxAndMin(cloud));

    vector<Point> p = findMaxAndMin(cloud);                                                               // 输入点云的最值点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_n(new pcl::PointCloud<pcl::PointXYZ>);                      // 存储归一化之后的点云
    cloud_n = normialize(cloud, p, "/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere**.ply"); // 归一化

    cout << "\nafter normalized:" << endl;
    // vector<Point> p_ = findMaxAndMin(cloud_n); // 归一化点云之后的最值点

    // 可视化
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    int v1(0);

    viewer.createViewPort(0, 0, 1, 1, v1);
    viewer.setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(cloud, 0, 0, 255);
    viewer.addPointCloud(cloud, point_cloud_color_handler, "cloud", v1);

    viewer.createViewPort(0, 0, 1, 1, v1);
    viewer.setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color2(cloud_n, 0, 0, 255);
    viewer.addPointCloud(cloud_n, color2, "cloud_n", v1);
    viewer.addText("cloud_normalized", 0, 0, "cloud_n", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_n");
    viewer.addCoordinateSystem(10);
    while (!viewer.wasStopped())
    {
        viewer.spin();
    }
    pcl::io::savePLYFile<pcl::PointXYZ>("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere**.ply", *cloud_n);
    return 0;
}

vector<Point> findMaxAndMin(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    vector<Point> vector_back;
    Point max, min;
    max.x = cloud_in->points[0].x;
    max.y = cloud_in->points[0].y;
    max.z = cloud_in->points[0].z;

    min.x = cloud_in->points[0].x;
    min.y = cloud_in->points[0].y;
    min.z = cloud_in->points[0].z;

    for (int i = 0; i < cloud_in->size(); i++)
    {
        if (cloud_in->points[i].x >= max.x)
        {
            max.x = cloud_in->points[i].x;
        }
        if (cloud_in->points[i].x <= min.x)
        {
            min.x = cloud_in->points[i].x;
        }

        if (cloud_in->points[i].y >= max.y)
        {
            max.y = cloud_in->points[i].y;
        }
        if (cloud_in->points[i].y <= min.y)
        {
            min.y = cloud_in->points[i].y;
        }

        if (cloud_in->points[i].z >= max.z)
        {
            max.z = cloud_in->points[i].z;
        }
        if (cloud_in->points[i].z <= min.z)
        {
            min.z = cloud_in->points[i].z;
        }
    }

    vector_back.push_back(max);
    vector_back.push_back(min);

    cout << "max: " << max.x << "," << max.y << "," << max.z << endl;
    cout << "min: " << min.x << "," << min.y << "," << min.z << endl;

    return vector_back;
}
// 转换球心坐标
pcl::PointCloud<pcl::PointXYZ>::Ptr normialize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<Point> MaxAndMin, string filename)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_back(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_back->width = cloud->width;
    cloud_back->height = cloud->height;
    cloud_back->is_dense = cloud->is_dense;
    cloud_back->resize(cloud->size());
    Point max_bound, min_bound;
    max_bound = MaxAndMin[0];
    min_bound = MaxAndMin[1];

    typedef pcl::PointXYZ PointT;
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
    for (auto c : cloud->points)
    {
        center0.x += c.x;
        center0.y += c.y;
        center0.z += c.z;
    }
    center0.x /= cloud->points.size();
    center0.y /= cloud->points.size();
    center0.z /= cloud->points.size();

    for (auto c : cloud->points)
    {
        center1.x += c.x * c.x;
        center1.y += c.y * c.y;
        center1.z += c.z * c.z;
    }
    center1.x /= cloud->points.size();
    center1.y /= cloud->points.size();
    center1.z /= cloud->points.size();

    for (auto c : cloud->points)
    {
        center2.x += c.x * c.y;
        center2.y += c.x * c.z;
        center2.z += c.y * c.z;
    }
    center2.x /= cloud->points.size();
    center2.y /= cloud->points.size();
    center2.z /= cloud->points.size();
    for (auto c : cloud->points)
    {
        center3.x += c.x * c.x * c.x;
        center3.y += c.x * c.x * c.y;
        center3.z += c.x * c.x * c.z;
    }
    center3.x /= cloud->points.size();
    center3.y /= cloud->points.size();
    center3.z /= cloud->points.size();
    for (auto c : cloud->points)
    {
        center4.x += c.x * c.y * c.y;
        center4.y += c.x * c.z * c.z;
        center4.z += c.y * c.y * c.y;
    }
    center4.x /= cloud->points.size();
    center4.y /= cloud->points.size();
    center4.z /= cloud->points.size();
    for (auto c : cloud->points)
    {
        center5.x += c.y * c.y * c.z;
        center5.y += c.y * c.z * c.z;
        center5.z += c.z * c.z * c.z;
    }
    center5.x /= cloud->points.size();
    center5.y /= cloud->points.size();
    center5.z /= cloud->points.size();
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

    for (int i = 0; i < cloud_back->size(); i++)
    {
        cloud_back->points[i].x = cloud->points[i].x - center.x;
        cloud_back->points[i].y = cloud->points[i].y - center.y;
        cloud_back->points[i].z = cloud->points[i].z - center.z;
    }

    //     // scale based on x range
    double scalecoe = (max_bound.x - min_bound.x) / 2;
    for (int i = 0; i < cloud_back->size(); i++)
    {
        cloud_back->points[i].x = cloud_back->points[i].x;
        cloud_back->points[i].y = cloud_back->points[i].y;
        cloud_back->points[i].z = cloud_back->points[i].z;
    }
    ostringstream oss;
    oss << "normalized_" << filename;
    pcl::io::savePLYFileASCII(oss.str(), *cloud_back);
    cout << "Normalized!" << endl;
    cout << oss.str() << " Saved!" << endl;
    return cloud_back;
}
// 转换质心坐标
//  pcl::PointCloud<pcl::PointXYZ>::Ptr normialize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, vector<Point> MaxAndMin, string filename)
//  {
//      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_back(new pcl::PointCloud<pcl::PointXYZ>);
//      cloud_back->width = cloud_in->width;
//      cloud_back->height = cloud_in->height;
//      cloud_back->is_dense = cloud_in->is_dense;
//      cloud_back->resize(cloud_in->size());
//      Point max_bound, min_bound;
//      max_bound = MaxAndMin[0];
//      min_bound = MaxAndMin[1];
//      double middlex, middley, middlez;
//      middlex = (max_bound.x + min_bound.x) / 2;
//      middley = (max_bound.y + min_bound.y) / 2;
//      middlez = (max_bound.z + min_bound.z) / 2;

//     for (int i = 0; i < cloud_back->size(); i++)
//     {
//         cloud_back->points[i].x = cloud_in->points[i].x - middlex;
//         cloud_back->points[i].y = cloud_in->points[i].y - middley;
//         cloud_back->points[i].z = cloud_in->points[i].z - middlez;
//     }

//     // scale based on x range
//     double scalecoe = (max_bound.x - min_bound.x) / 2;
//     for (int i = 0; i < cloud_back->size(); i++)
//     {
//         //     cloud_back->points[i].x = cloud_back->points[i].x / scalecoe;
//         //     cloud_back->points[i].y = cloud_back->points[i].y / scalecoe;
//         //     cloud_back->points[i].z = cloud_back->points[i].z / scalecoe;
//         cloud_back->points[i].x = cloud_back->points[i].x;
//         cloud_back->points[i].y = cloud_back->points[i].y;
//         cloud_back->points[i].z = cloud_back->points[i].z;
//     }
//     //    pcl::io::savePCDFileASCII("normalized.pcd",*cloud_back);
//     ostringstream oss;
//     oss << "normalized_" << filename;
//     pcl::io::savePLYFileASCII(oss.str(), *cloud_back);
//     cout << "Normalized!" << endl;
//     cout << oss.str() << " Saved!" << endl;
//     return cloud_back;
// }