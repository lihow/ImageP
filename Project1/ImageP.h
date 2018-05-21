#ifndef IMAGEP_H
#define IMAGEP_H

//��׼��
#include<iostream>
#include <vector>
#include <string>
#include <fstream> 
using namespace std;
//opencvͷ�ļ�
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;
//OCRͷ�ļ�
#include "OCR_interface.h"

//�궨��
#define PI 3.1415926

class ImageP{
public:
	Mat FindDiff(const string PicPath, bool show = false, const string OutPath = "C:/Users/Lenovo/Desktop/FindDiff.jpg");
	Mat SiftMatch(const string PicPath_1, const string PicPath_2, const string OutPath = "C:/Users/Lenovo/Desktop/SiftMatch.jpg", bool show = false);
	Mat SurfFea(const string PicPath, bool show = true);
	Mat	Gradient(const string PicPath, bool show = true);
	Mat HogPeople(const string PicPath, bool show = false);
	Mat LoGOperator(const string PicPath, bool show = true);
	Mat LBP(const string PicPath, bool show = true);
	Mat Histogram1D(const Mat &image);
	Mat LaplacePyramid(const string PicPath, int levels = 2, bool show = true);
	Mat AddSaltNoise(const string PicPath, int n = 3000, bool show = true);
	void Blur(const Mat &Image, bool show = true);
	Mat LineFind(const string PicPath, bool show = true);
	Mat BackgroundTransfer(const string PicPath, bool show = true);
	//��ʾ����λ��
	void PiexLocation_Show(const string PicPath);
	//void PiexLocation_Mouse(int EVENT, int x, int y, int flags, void* userdata);
	/*****************************************ú̿���*****************************/
	Mat FrontSeg(const string PicPath, bool show = true);
	double CountWdith(const string refFramePath, const string curFramePath, bool show = true);
	//��������
	Mat SlidingWnd(Mat &src, vector<Mat> &wnd, int n = 6/*, Size &wndSize, double x_percent, double y_percent*/);
	//����ƽ���ݶ�
	double CalMeanGrad(Mat img);
	//Mat FaceDetect()
	/*******************************************����ʦOCR*************************************/
	string VilabOCR(const string PicPath, bool show = true);
	Mat RemoveLine(Mat src, bool show = true);
	//ģ��ƥ��
	//������ƥ��
	void SingleTemplateMatch(const string  PicPath, const string TemplPath, bool show = true);
	void MultiTemplateMatch(const string  PicPath, const string TemplPath, bool show = true);
	
	/**********************************ֽ��ʶ��**********************************************/
	void GetContoursPic(const string pSrcFileName, const string pDstFileName);
	Mat MoneyROI(const string PicPath, bool show = true);
	Rect GroupRect(vector<Rect>RectList);
private:
	//LBP ����
	int radius = 1;
	int neighbors = 8;
	//һάֱ��ͼ����
	int histsize[1];
	float hranges[2];
	const float *ranges[1];
	int channels[1];
};

#endif