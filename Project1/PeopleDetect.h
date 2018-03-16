#ifndef __PEOPLEDETECT_H__
#define __PEOPLEDETECT_H__
#include <iostream>
#include <opencv.hpp>
using namespace std;
using namespace cv;

class PeoDetect{

private:
	vector<Rect> peoples;
	CascadeClassifier cascade;

public:
	vector<Rect> detectPeople(Mat &src);

	PeoDetect(){
		cascade.load("D:\\opencv2.4.10\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml");
	}
	~PeoDetect(){
	}
};

#endif