#ifndef IMAGEP_H
#define IMAGEP_H

//标准库
#include<iostream>
#include <vector>
#include <string>
#include <fstream> 
using namespace std;
//opencv头文件
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;
//OCR头文件
#include "OCR_interface.h"

//宏定义
#define PI 3.1415926

class ImageP{
public:
	Mat FindDiff(const string PicPath, bool show = false, const string OutPath = "C:/Users/Lenovo/Desktop/FindDiff.jpg");
	Mat SiftMatch(const string PicPath_1, const string PicPath_2, const string OutPath = "C:/Users/Lenovo/Desktop/SiftMatch.jpg", bool show = false);
	Mat SurfFea(const string PicPath, bool show = true);
	Mat	Gradient(const string PicPath, bool show = true);
	Mat HogPeople(const string PicPath, bool show = false);
	Mat LoGOperator(const string PicPath, bool show = true);
	Mat LBP(Mat img, bool show = true);
	Mat Histogram1D(const Mat &image);
	Mat LaplacePyramid(const string PicPath, int levels = 2, bool show = true);
	Mat AddSaltNoise(const string PicPath, int n = 3000, bool show = true);
	void Blur(const Mat &Image, bool show = true);
	Mat LineFind(const string PicPath, bool show = true);
	Mat BackgroundTransfer(const string PicPath, bool show = true);
	//显示像素位置
	void PiexLocation_Show(/*const string PicPath*/Mat src);
	//void PiexLocation_Mouse(int EVENT, int x, int y, int flags, void* userdata);
	/*****************************************煤炭检测*****************************/
	Mat FrontSeg(const string PicPath, bool show = true);
	double CountWdith(const string refFramePath, const string curFramePath, bool show = true);
	Mat ColHistogram(Mat src);//1D纵向灰度直方图
	double CountMean(Mat src);//计算灰度图的均值
	Mat BlockTest(Mat src, Mat characImg);//分割小块测试
	Mat get_perspective_mat();//透视变换
	//滑动窗口
	Mat SlidingWnd(Mat &src, vector<Mat> &wnd, int n = 6/*, Size &wndSize, double x_percent, double y_percent*/);
	//计算平均梯度
	double CalMeanGrad(Mat img);
	//Mat FaceDetect()
	/*******************************************刘老师OCR*************************************/
	string VilabOCR(const string PicPath, bool show = true);
	Mat RemoveLine(Mat src, bool show = true);
	//模板匹配
	//单对象匹配
	void SingleTemplateMatch(const string  PicPath, const string TemplPath, bool show = true);
	void MultiTemplateMatch(const string  PicPath, const string TemplPath, bool show = true);
	
	/**********************************纸币识别**********************************************/
	void GetContoursPic(const string pSrcFileName, const string pDstFileName);
	Mat MoneyROI(const string PicPath, bool show = true);
	Rect GroupRect(vector<Rect>RectList);
private:
	//LBP 参数
	int radius = 1;
	int neighbors = 8;
	//一维直方图参数
	int histsize[1];
	float hranges[2];
	const float *ranges[1];
	int channels[1];
};

#endif