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
using namespace cv;
//OCR头文件
#include "OCR_interface.h"

//宏定义
#define PI 3.1415926

class ImageP{
public:
	Mat FindDiff(const string PicPath, bool show = false, const string OutPath = "C:/Users/Lenovo/Desktop/FindDiff.jpg");
	Mat SiftMatch(const string PicPath_1, const string PicPath_2, const string OutPath = "C:/Users/Lenovo/Desktop/SiftMatch.jpg", bool show = false);
	Mat SurfFea(const string PicPath, bool show = false);
	Mat HogPeople(const string PicPath, bool show = false);
	Mat LoGOperator(const string PicPath, bool show = true);
	Mat LBP(const string PicPath, bool show = true);
	Mat Histogram1D(const Mat &image);
	Mat LaplacePyramid(const string PicPath, int levels = 2, bool show = true);
	Mat AddSaltNoise(const string PicPath, int n = 3000, bool show = true);
	void Blur(const Mat &Image, bool show = true);
	Mat LineFind(const string PicPath, bool show = true);
	Mat BackgroundTransfer(const string PicPath, bool show = true);
	//煤炭检测
	Mat FrontSeg(const string PicPath, bool show = true);
	double CountWdith(const string refFramePath, const string curFramePath, bool show = true);
	//Mat FaceDetect()
	//刘老师OCR
	string VilabOCR(const string PicPath, bool show = true);
	Mat RemoveLine(const string PicPath, bool show = true);
	//纸币识别
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