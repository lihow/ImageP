#include "ImageP.h"
#include "HogSVM.h"
#include "VideoP.h"


void compute_absolute_mat(const Mat& in, Mat & out)
{
	if (out.empty()){
		out.create(in.size(), CV_32FC1);
	}

	const Mat_<Vec2f> _in = in;
	//遍历吧，少年  
	for (int i = 0; i < in.rows; ++i){
		float *data = out.ptr<float>(i);
		for (int j = 0; j < in.cols; ++j){
			double s = _in(i, j)[0] * _in(i, j)[0] + _in(i, j)[1] * _in(i, j)[1];
			if (s>1){
				data[j] = std::sqrt(s);
			}
			else{
				data[j] = 0.0;
			}

		}
	}
}


int main(){

	ImageP Processor;
	string PicPath1 = "C:\\Users\\Lenovo\\Desktop\\QQ截图20180520143811.png";
	Mat img = imread(PicPath1);

// 	vector<Mat>wnd;
//	Mat dst = Processor.SlidingWnd(img, wnd,10);
//	imshow("src", dst);
//	waitKey(0);

	string videoPath = "F:\\煤炭识别\\煤炭视频\\煤炭视频3\\哈M13机头入料口 20180417 174000 20180417 204030.avi";
	string videoPath1 = "F:\\煤炭识别\\煤炭视频\\煤炭视频1\\Mei-part.avi";



	/*********************************三帧差法检测物体********************************************/
	
	VideoCapture videoCap(videoPath);
	if (!videoCap.isOpened())
	{
		return -1;
	}
	double videoFPS = videoCap.get(CV_CAP_PROP_FPS);  //获取帧率  
	double videoPause = 1000 / videoFPS;
	Mat framePrePre; //上上一帧  
	Mat framePre; //上一帧  
	Mat frameNow; //当前帧  
	Mat frameDet; //运动物体  
	videoCap >> framePrePre;
	videoCap >> framePre;
	cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);
	cvtColor(framePre, framePre, CV_RGB2GRAY);
	int save = 0;
	while (true)
	{
		videoCap >> frameNow;
		if (frameNow.empty() || waitKey(videoPause) == 27)
		{
			break;
		}
		cvtColor(frameNow, frameNow, CV_RGB2GRAY);
		Mat Det1;
		Mat Det2;
		absdiff(framePrePre, framePre, Det1);  //帧差1  
		absdiff(framePre, frameNow, Det2);     //帧差2  
		threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //自适应阈值化  
		threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);
		Mat element = getStructuringElement(0, Size(3, 3));  //膨胀核  
		dilate(Det1, Det1, element);    //膨胀  
		dilate(Det2, Det2, element);
		bitwise_and(Det1, Det2, frameDet);
		framePrePre = framePre;
		framePre = frameNow;
		imshow("Video", frameNow);
		imshow("Detection", frameDet);
	}


	




	/******************************************基于光流法******************************************************/
	/*
	//方法一 寻找特征点
	Mat image1, image2;
	vector<Point2f> point1, point2, pointCopy;
	vector<uchar> status;
	vector<float> err;
	VideoCapture video(videoPath1);
	double fps = video.get(CV_CAP_PROP_FPS); //获取视频帧率    
	double pauseTime = 1000 / fps; //两幅画面中间间隔     
	video >> image1;
	Mat image1Gray, image2Gray;
	cvtColor(image1, image1Gray, CV_RGB2GRAY);
	goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
	pointCopy = point1;
	for (int i = 0; i < point1.size(); i++)    //绘制特征点位  
	{
		circle(image1, point1[i], 1, Scalar(0, 0, 255), 2);
	}
	namedWindow("角点特征光流", 0);
	imshow("角点特征光流", image1);
	while (true)
	{
		video >> image2;
		if (!image2.data || waitKey(pauseTime) == 27)  //图像为空或Esc键按下退出播放    
		{
			break;
		}
		cvtColor(image2, image2Gray, CV_RGB2GRAY);
		calcOpticalFlowPyrLK(image1Gray, image2Gray, point1, point2, status, err, Size(20, 20), 3); //LK金字塔       
		for (int i = 0; i < point2.size(); i++)
		{
			circle(image2, point2[i], 1, Scalar(0, 0, 255), 2);
			line(image2, pointCopy[i], point2[i], Scalar(255, 0, 0), 2);
		}
		imshow("角点特征光流", image2);
		swap(point1, point2);
		image1Gray = image2Gray.clone();
	}
	*/

	/*
	//方法二 
	VideoCapture cap;
	cap.open(videoPath1);

	if (!cap.isOpened()){
		std::cout << "视频读取失败！" << std::endl;
		return -1;
	}

	Mat  gray, prvGray, optFlow, absoluteFlow, img_for_show;
	while (1){
		cap >> img;
		if (img.empty()) break;

		cvtColor(img, gray, CV_BGR2GRAY);
		if (prvGray.data){
			calcOpticalFlowFarneback(prvGray, gray, optFlow, 0.5, 3, 15, 3, 5, 1.2, 0); //使用论文参数  
			compute_absolute_mat(optFlow, absoluteFlow);
			normalize(absoluteFlow, img_for_show, 0, 255, NORM_MINMAX, CV_8UC1);
			
			imshow("opticalFlow", img_for_show);
			imshow("resource", img);
		}
		cv::swap(prvGray, gray);

		waitKey(1);
	}

	*/












	//sobel算子
	Mat grad_x, abs_grad_x;
	//Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);


//	imshow("result", abs_grad_x);
//	waitKey();



	VideoP player;
	//player.PlayVideo(videoPath);
	//player.VideoBackgroundSubtractor(videoPath1);
	getchar();
	return 0;
}