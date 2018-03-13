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
using namespace cv;

//�궨��
#define PI 3.1415926

class ImageP{
public:
	Mat FindDiff(const string PicPath, const string OutPath = "C:/Users/Lenovo/Desktop/FindDiff.jpg", bool show = false);
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
	//ú̿���
	Mat FrontSeg(const string PicPath, bool show = true);
	double CountWdith(const string refFramePath, const string curFramePath, bool show = true);
	//Mat FaceDetect()
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