#include "../inc/opencv.h"
int main()

{
	cv::Mat input, color, gray, hsv, median, gauss, average, dilate, erode, harrisNorm, harris, canny;
	input = cv::imread("../images/sea.png");
	if (input.empty())
	{
		std::cout << "Can not read image" << std::endl;
		return 0;
	}
	color = input;

	//
	cv::imshow("color", color);
	cv::waitKey(0);
	return 0;
	//
	cv::imwrite("color", color);
	//

	cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);
	cv::cvtColor(color, hsv, cv::COLOR_RGB2HSV);
	imshow("gray", gray);
	imshow("hsv", hsv);
	//
	cv::GaussianBlur(color, gauss, cv::Size(5, 5), 1, 1);
	cv::medianBlur(color, median, 5);
	cv::blur(color, average, cv::Size(3, 3), cv::Point(-1, -1));

	cv::imshow("blur", average);
	cv::imshow("gauss", gauss);
	cv::imshow("median", median);

	//
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(input, dilate, element);
	cv::erode(input, erode, element);
	cv::imshow("dilate", dilate);
	cv::imshow("erode", erode);

	//
	cv::Canny(gray, canny, 10, 100);
	cv::imshow("canny", canny);
	//
	cv::cornerHarris(gray, harris, 2, 3, 0.04);
	cv::normalize(harris, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	for (size_t row = 0; row < color.rows; row++)
	{
		for (size_t col = 0; col < color.cols; col++)
		{
			int rsp = harrisNorm.at<uchar>(row, col);
			if (rsp > 150)
				cv::circle(color, cv::Point(row, col), 5, cv::Scalar(0, 0, 255), 1);
			// cv::circle(color,cv::Point(col,row),3, cv::Scalar(255,10,10),1);
		}
	}
}
