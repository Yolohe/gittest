#include "../inc/CloudCompare.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;

void print4x4Matrix(const Eigen::Matrix4d &matrix)
{
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

// void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *nothing)
// {
//     if (event.getKeySym() == "space" && event.keyDown())
//     {
//         next_iteration = true;
//     }
// }

int main(int argc, char *argv[])
{
    PointCloudT::Ptr cloud_in(new PointCloudT);  // 初始点云
    PointCloudT::Ptr cloud_tr(new PointCloudT);  // 转换点云
    PointCloudT::Ptr cloud_icp(new PointCloudT); // 输出点云
    pcl::io::loadPLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere21*.ply", *cloud_in);

    int iterations = 1; // 默认ICP配准的迭代次数
    if (argc > 2)
    {

        iterations = atoi(argv[2]); // 将字符串变量转换为整数变量
        if (iterations < 1)
        {
            PCL_ERROR("Number of initial iterations must be >= 1\n");
            return (-1);
        }
    }

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity(); // 定义旋转矩阵和平移矩阵

    /* 旋转矩阵的具体定义 */
    double theta = M_PI / 20; // 设置旋转弧度的角度
    transformation_matrix(0, 0) = cos(theta);
    transformation_matrix(0, 1) = -sin(theta);
    transformation_matrix(1, 0) = sin(theta);
    transformation_matrix(1, 1) = cos(theta);

    /* 设置平移矩阵 */
    transformation_matrix(0, 3) = 0.0;
    transformation_matrix(1, 3) = 0.0;
    transformation_matrix(2, 3) = 0.0;

    /* 在终端输出转换矩阵 */
    // std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
    // print4x4Matrix(transformation_matrix);

    // 执行初始变换
    pcl::transformPointCloud(*cloud_in, *cloud_icp, transformation_matrix);
    *cloud_tr = *cloud_icp; // 将cloud_icp变量备份

    /* 设置ICP配准算法输入参数 */

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(iterations); // 设置迭代次数
    icp.setInputSource(cloud_icp);
    icp.setInputTarget(cloud_in);
    icp.align(*cloud_icp);
    icp.setMaximumIterations(1); // 当再次调用.align ()函数时，我们设置该变量为1。

    if (icp.hasConverged())
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        // print4x4Matrix(transformation_matrix);
    }
    else
    {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }

    pcl::visualization::PCLVisualizer viewer("ICP demo"); // 初始可视化对象
    int v1(0);                                            // 创建视口v1
    int v2(1);                                            // 创建视口v2
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer.addPointCloud(cloud_in, "cloud_in");

    float bckgr_gray_level = 0.0; // 黑色
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    /* 设置初始点云为白色，并在左右视口都显示 */
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v2", v2);

    /* 设置用构造的变换矩阵变换后的点云为绿色 */
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(cloud_tr, 0, 250, 0);
    viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "cloud_tr_v1", v1);

    /* 设置ICP配准后的点云为红色 */
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icp, 250, 0, 0);
    viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

    std::stringstream ss;
    ss << iterations;
    std::string iterations_cnt = "ICP iterations = " + ss.str();
    viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

    /* 设置背景颜色 */
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    /* 通过键盘，调用回调函数 */
    // viewer.registerKeyboardCallback(&keyboardEventOccurred, (void *)NULL);

    /* 设置显示器 */
    while (!viewer.wasStopped())
    {
        viewer.spin();

        if (next_iteration) // 用户按下空格键
        {
            // 配准算法开始迭代
            icp.align(*cloud_icp);

            if (icp.hasConverged())
            {
                printf("\033[11A");
                printf("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
                std::cout << "\nICP transformation " << ++iterations << " : cloud_icp -> cloud_in" << std::endl;
                transformation_matrix *= icp.getFinalTransformation().cast<double>();
                // print4x4Matrix(transformation_matrix); // 输出初始矩阵和当前矩阵的转换矩阵

                ss.str("");
                ss << iterations;
                std::string iterations_cnt = "ICP iterations = " + ss.str();
                viewer.updateText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
                viewer.updatePointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2");
            }
            else
            {
                PCL_ERROR("\nICP has not converged.\n");
                return (-1);
            }
        }
        next_iteration = false;
    }
    return (0);
    pcl::io::savePLYFile("/home/ddxy/Downloads/视觉部分/kinect/camera/testcloud/sphere21*.ply", *cloud_icp);
}
