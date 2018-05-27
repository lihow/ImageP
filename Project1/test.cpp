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

/****************以下是三帧差法*********************************/
Mat image;   //读入视频帧的Mat  
char* windowName = "Video Control"; //播放窗口名称  
char* trackBarName = "播放进度";    //trackbar控制条名称  
double totalFrame = 1.0;     //视频总帧数  
double currentFrame = 1.0;    //当前播放帧  
int trackbarValue = 1;    //trackbar控制量  
int trackbarMax = 1000;   //trackbar控制条最大值  
double frameRate = 1.0;  //视频帧率  
VideoCapture video;    //声明视频对象  
double controlRate = 0.1;

void TrackBarFunc(int, void(*))
{
	controlRate = (double)trackbarValue / trackbarMax*totalFrame; //trackbar控制条对视频播放进度的控制  
	video.set(CV_CAP_PROP_POS_FRAMES, controlRate);   //设置当前播放帧  
}

/*********************圈出多边形区域*********************************/

//Mat image;
Mat imageCopy; //绘制矩形框时用来拷贝原图的图像  
bool leftButtonDownFlag = false; //左键单击后视频暂停播放的标志位  
Point originalPoint; //矩形框起点  
Point processPoint; //矩形框终点 
bool subFlag = false;
void onMouse(int event, int x, int y, int flags, void *ustc)
{

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		leftButtonDownFlag = true; //标志位  
		originalPoint = Point(x, y);  //设置左键按下点的矩形起点  
		processPoint = originalPoint;
	}
	if (event == CV_EVENT_MOUSEMOVE&&leftButtonDownFlag)
	{
		processPoint = Point(x, y);
	}
	if (event == CV_EVENT_LBUTTONUP || event == CV_EVENT_LBUTTONUP)
	{
		leftButtonDownFlag = false;
		//subFlag = true;
		//Mat rectImage = image(Rect(originalPoint, processPoint)); //子图像显示  
		//imshow("Sub Image", rectImage);
	}
}


int main(){

	ImageP Processor;
	string PicPath1 = "C:\\Users\\Lenovo\\Desktop\\p1.png";
	Mat img = imread(PicPath1, 0);

	//Processor.PiexLocation_Show(img);

	string videoPath = "F:\\煤炭识别\\煤炭视频\\煤炭视频5\\ch01_20180523175108.mp4";
	string videoPath1 = "F:\\煤炭识别\\煤炭视频\\煤炭视频1\\Mei-total.avi";

	//video.open(videoPath);
	//Mat frameROI;
	//video >> frameROI;
	//cvtColor(frameROI, frameROI, CV_RGB2GRAY);

	//Processor.PiexLocation_Show(frameROI);


	/*********************************三帧差法检测物体********************************************/

	//改进
	video.open(videoPath);//鼠标控制视频需要设置全局变量
	//VideoCapture video(videoPath);
	if (!video.isOpened())
	{
		return -1;
	}
	totalFrame = video.get(CV_CAP_PROP_FRAME_COUNT);  //获取总帧数  
	double videoFPS = video.get(CV_CAP_PROP_FPS);  //获取帧率  
	double videoPause = 1000 / videoFPS;


	namedWindow(windowName);
	createTrackbar(trackBarName, windowName, &trackbarValue, trackbarMax, TrackBarFunc);
	TrackBarFunc(0, 0);

	Mat framePrePre; //上上一帧  
	Mat framePre; //上一帧  
	Mat frameNow; //当前帧  
	Mat frameDet; //运动物体  
	Mat frameOri; //未被处理的原始帧
	video >> framePrePre;

	//分割出ROI
	cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);

	Mat roi = Mat::zeros(framePrePre.rows, framePrePre.cols, CV_8UC1);
	//vector<Point>pts = { Point(574, 7), Point(332, 708), Point(1274, 709), Point(1274, 579), Point(794, 7) };//场景1
	vector<Point>pts = { Point(338, 6), Point(30, 571), Point(644, 570), Point(502, 9) };//场景2
	//ruliaokou
	//pts.push_back(Point(741, 331));
	//pts.push_back(Point(126, 711));
	//pts.push_back(Point(866, 715));
	//pts.push_back(Point(775, 334));

	//vector<vector<Point>>contour = { pts };
	//drawContours(roi, contour, 0, Scalar::all(255), CV_FILLED);//掩模

	video >> framePre;
	//cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);
	cvtColor(framePre, framePre, CV_RGB2GRAY);
	int save = 0;
	while (true)
	{
		video >> frameNow;
		frameNow.copyTo(frameOri);
		Mat grad_y, abs_grad_y;

		if (frameNow.empty() || waitKey(videoPause) == 27)
		{
			break;
		}
		cvtColor(frameNow, frameNow, CV_RGB2GRAY);
		Mat frameNow1;
		frameNow.copyTo(frameNow1);

		//bitwise_and(frameNow, roi, frameNow);

		Mat Det1;
		Mat Det2;
		absdiff(framePrePre, framePre, Det1);  //帧差1  
		absdiff(framePre, frameNow, Det2);     //帧差2  
		Mat meanMat, devMat;
		//待调整
		meanStdDev(Det1, meanMat, devMat);
		if (meanMat.at<double>(0, 0) >1 && devMat.at<double>(0, 0) >0.8)
			threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //自适应阈值化  
		meanStdDev(Det2, meanMat, devMat);
		if (meanMat.at<double>(0, 0) >1 && devMat.at<double>(0, 0) >0.8)
			threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);
		Mat element = getStructuringElement(0, Size(3, 3));  //膨胀核  
		dilate(Det1, Det1, element);    //膨胀  
		dilate(Det2, Det2, element);
		bitwise_and(Det1, Det2, frameDet);
		framePrePre = framePre;
		framePre = frameNow;


		//分块测试
		Mat frameDet1 = Processor.BlockTest(frameNow1, frameDet);

		resize(frameOri, frameOri, Size(650, 550));
		resize(frameDet, frameDet, Size(650, 550));
		resize(frameDet1, frameDet1, Size(650, 550));
		imshow(windowName, frameOri);
		imshow("Detection", frameDet);
		imshow("BlockTest", frameDet1);
		imshow("ColHistogram", Processor.ColHistogram(frameNow1));
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

	//
	////方法二 
	//VideoCapture cap;
	//cap.open(videoPath1);

	//if (!cap.isOpened()){
	//	std::cout << "视频读取失败！" << std::endl;
	//	return -1;
	//}

	//Mat  gray, prvGray, optFlow, absoluteFlow, img_for_show;
	//while (1){
	//	cap >> img;
	//	if (img.empty()) break;

	//	cvtColor(img, gray, CV_BGR2GRAY);
	//	if (prvGray.data){
	//		calcOpticalFlowFarneback(prvGray, gray, optFlow, 0.5, 3, 15, 3, 5, 1.2, 0); //使用论文参数  
	//		compute_absolute_mat(optFlow, absoluteFlow);
	//		normalize(absoluteFlow, img_for_show, 0, 255, NORM_MINMAX, CV_8UC1);
	//		imshow("opticalFlow", img_for_show);
	//		//imshow("opticalFlow", absoluteFlow);
	//		imshow("resource", img);
	//	}
	//	cv::swap(prvGray, gray);

	//	waitKey(1);
	//}

	

	/******************************基于方差*************************************/
	//原始方法
/*
	int n = 50;
	video.open(videoPath);
	if (!video.isOpened())
	{
		return -1;
	}
	namedWindow(windowName);
	createTrackbar(trackBarName, windowName, &trackbarValue, trackbarMax, TrackBarFunc);
	TrackBarFunc(0, 0);
	totalFrame = video.get(CV_CAP_PROP_FRAME_COUNT);  //获取总帧数 
	double videoFPS = video.get(CV_CAP_PROP_FPS);  //获取帧率  
	double videoPause = 1000 / videoFPS;
	Mat frame, origin, result;


	video >> frame;;
	cvtColor(frame, frame, CV_RGB2GRAY);

	//分割出ROI
	vector<vector<Point>>contour;
	vector<Point>pts;
	Mat roi = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	//Mei-part

	pts.push_back(Point(221, 16));
	pts.push_back(Point(7, 346));
	pts.push_back(Point(10, 561));
	pts.push_back(Point(917, 562));
	pts.push_back(Point(647, 24));


	//ruliaokou
	pts.push_back(Point(741, 331));
	pts.push_back(Point(126, 711));
	pts.push_back(Point(866, 715));
	pts.push_back(Point(775, 334));

	contour.push_back(pts);
	drawContours(roi, contour, 0, Scalar::all(255), CV_FILLED);
	//bitwise_and(frame, roi, frame);


//	videoCap >> frame;
//	cvtColor(frame, frame, CV_RGB2GRAY);

	while (true)
	{
		video >> frame;
		if (frame.empty() || waitKey(videoPause) == 27)
		{
			break;
		}
		origin = frame.clone();//原始
		result = frame.clone();//结果
		cvtColor(frame, frame, CV_RGB2GRAY);

		//提取ROI
		bitwise_and(frame, roi, frame);


		//原图上画线
		for (int i = 0; i < frame.cols - frame.cols / n + 1; i += frame.cols / n){//竖线
			Point start = Point(i, 0);
			Point end = Point(i, frame.rows);
			line(result, start, end, Scalar(0, 255, 0));
		}
		for (int j = 0; j < frame.rows - frame.rows / n + 1; j += frame.rows / n){//横线
			Point start = Point(0, j);
			Point end = Point(frame.cols, j);
			line(result, start, end, Scalar(0, 255, 0));
		}
		Mat Mean, Stddv;//均值和方差

		
		Mat mask = Mat::zeros(n, n, CV_8UC1);
		//先列后行
//		for (int i = 0, im = 0; i < frame.cols - frame.cols / n + 1; i += frame.cols / n, im++){
//			for (int j = 0, jm = 0; j < frame.rows - frame.rows / n + 1; j += frame.rows / n, jm++){
		for (int j = 0, im = 0; j < frame.rows - frame.rows / n + 1 && im < n; j += frame.rows / n, im++){
			for (int i = 0, jm = 0; i < frame.cols - frame.cols / n + 1 && jm < n; i += frame.cols / n, jm++){
			
				Mat ROI = frame(Rect(i, j, frame.cols / n, frame.rows / n));

				//计算区域均值和方差
				meanStdDev(ROI, Mean, Stddv);
				char meanStr[10];
				char stddvStr[10];
				string str = to_string(int(Stddv.at<double>(0, 0)));
				mask.at<uchar>(im, jm) = int(Stddv.at<double>(0, 0));
				threshold(mask, mask, 5, 255, THRESH_BINARY);
				putText(result, str, Point(i + frame.cols / (2 * n), j + frame.rows / (2 * n)), cv::FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
			}
		}

		resize(origin, origin, Size(650, 550));
		//resize(result, result, Size(650, 550));
		imshow(windowName, result);
		//imshow("Detection", result);
		resize(mask, mask, Size(650, 550));
		imshow("Detection", mask);
	}
*/








	//sobel算子
	Mat grad_x, abs_grad_x;
	//Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);


//	imshow("result", abs_grad_x);
//	waitKey();



	VideoP player;
	//player.PlayVideo(videoPath1);
	//player.VideoBackgroundSubtractor(videoPath1);
	getchar();
	return 0;
}