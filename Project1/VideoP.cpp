#include "VideoP.h"
/*
@function:读取本地摄像头
*/
void VideoP::LocalCamera(){
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return;
	Mat inframe, outframe;
	while (1){
		cap >> inframe;
		if (inframe.empty())
			break;
		imshow("inputCamera", inframe);

		if (inframe.channels() == 3){
			cvtColor(inframe, outframe, CV_BGR2GRAY);
		}
		Canny(outframe, outframe, 100, 120);
		threshold(outframe, outframe, 50, 255, THRESH_BINARY_INV);
		imshow("outputCamera", outframe);
		if (waitKey(20) > 0)
			break;
	}
	cap.release();
	destroyAllWindows();
}
/*
@function: 基于高斯混合模型GMM的前景/背景分割算法用于动态物体检测
@param VideoPath: 视频路径
*/
void VideoP::VideoBackgroundSubtractor(const string VideoPath){
	VideoCapture video(VideoPath);
	if (!video.isOpened()){
		cout << "fail to open!" << endl;
		return;
	}
	
	int frameNum = 1;
	long totalFrameNumber = video.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame, mask, thresholdImage, output;
	video >> frame;
	BackgroundSubtractorMOG bgSubtractor(20, 10, 0.5, false);
	while (true){
		if (totalFrameNumber == frameNum)
			break;
		video >> frame;
		++frameNum;
		bgSubtractor(frame, mask, 0.001);
		imshow("mask", mask);
		waitKey(10);
	}
}