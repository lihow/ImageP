#include "ImageP.h"
#include "HogSVM.h"
#include "VideoP.h"


void compute_absolute_mat(const Mat& in, Mat & out)
{
	if (out.empty()){
		out.create(in.size(), CV_32FC1);
	}

	const Mat_<Vec2f> _in = in;
	//�����ɣ�����  
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

/****************��������֡�*********************************/
Mat image;   //������Ƶ֡��Mat  
char* windowName = "Video Control"; //���Ŵ�������  
char* trackBarName = "���Ž���";    //trackbar����������  
double totalFrame = 1.0;     //��Ƶ��֡��  
double currentFrame = 1.0;    //��ǰ����֡  
int trackbarValue = 1;    //trackbar������  
int trackbarMax = 255;   //trackbar���������ֵ  
double frameRate = 1.0;  //��Ƶ֡��  
VideoCapture video;    //������Ƶ����  
double controlRate = 0.1;

void TrackBarFunc(int, void(*))
{
	controlRate = (double)trackbarValue / trackbarMax*totalFrame; //trackbar����������Ƶ���Ž��ȵĿ���  
	video.set(CV_CAP_PROP_POS_FRAMES, controlRate);   //���õ�ǰ����֡  
}

/*********************Ȧ�����������*********************************/

//Mat image;
Mat imageCopy; //���ƾ��ο�ʱ��������ԭͼ��ͼ��  
bool leftButtonDownFlag = false; //�����������Ƶ��ͣ���ŵı�־λ  
Point originalPoint; //���ο����  
Point processPoint; //���ο��յ� 
bool subFlag = false;
void onMouse(int event, int x, int y, int flags, void *ustc)
{

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		leftButtonDownFlag = true; //��־λ  
		originalPoint = Point(x, y);  //����������µ�ľ������  
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
		//Mat rectImage = image(Rect(originalPoint, processPoint)); //��ͼ����ʾ  
		//imshow("Sub Image", rectImage);
	}
}


int main(){

	ImageP Processor;
	string PicPath1 = "C:\\Users\\Lenovo\\Desktop\\QQ��ͼ20180520143811.png";
	Mat img = imread(PicPath1);

// 	vector<Mat>wnd;
//	Mat dst = Processor.SlidingWnd(img, wnd,10);
//	imshow("src", dst);
//	waitKey(0);

	string videoPath = "F:\\ú̿ʶ��\\ú̿��Ƶ\\ú̿��Ƶ3\\ruliaokou-part.avi";
	string videoPath1 = "F:\\ú̿ʶ��\\ú̿��Ƶ\\ú̿��Ƶ1\\Mei-part.avi";



	/*********************************��֡��������********************************************/
	
	//ԭʼ����
	//VideoCapture videoCap(videoPath);
	//if (!videoCap.isOpened())
	//{
	//	return -1;
	//}
	//double videoFPS = videoCap.get(CV_CAP_PROP_FPS);  //��ȡ֡��  
	//double videoPause = 1000 / videoFPS;
	//Mat framePrePre; //����һ֡  
	//Mat framePre; //��һ֡  
	//Mat frameNow; //��ǰ֡  
	//Mat frameDet; //�˶�����  
	//videoCap >> framePrePre;
	//videoCap >> framePre;
	//cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);
	//cvtColor(framePre, framePre, CV_RGB2GRAY);
	//int save = 0;
	//while (true)
	//{
	//	videoCap >> frameNow;
	//	if (frameNow.empty() || waitKey(videoPause) == 27)
	//	{
	//		break;
	//	}
	//	cvtColor(frameNow, frameNow, CV_RGB2GRAY);
	//	Mat Det1;
	//	Mat Det2;
	//	absdiff(framePrePre, framePre, Det1);  //֡��1  
	//	absdiff(framePre, frameNow, Det2);     //֡��2  
	//	threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //����Ӧ��ֵ��  
	//	threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);
	//	Mat element = getStructuringElement(0, Size(3, 3));  //���ͺ�  
	//	dilate(Det1, Det1, element);    //����  
	//	dilate(Det2, Det2, element);
	//	bitwise_and(Det1, Det2, frameDet);
	//	framePrePre = framePre;
	//	framePre = frameNow;
	//	imshow("Video", frameNow);
	//	imshow("Detection", frameDet);
	//}

	//�Ľ�
	//video.open(videoPath1);
	////VideoCapture video(videoPath);
	//if (!video.isOpened())
	//{
	//	return -1;
	//}
	//totalFrame = video.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ��֡��  
	//double videoFPS = video.get(CV_CAP_PROP_FPS);  //��ȡ֡��  
	//double videoPause = 1000 / videoFPS;


	//namedWindow(windowName);
	//createTrackbar(trackBarName, windowName, &trackbarValue, trackbarMax, TrackBarFunc);
	//TrackBarFunc(0, 0);

	//Mat framePrePre; //����һ֡  
	//Mat framePre; //��һ֡  
	//Mat frameNow; //��ǰ֡  
	//Mat frameDet; //�˶�����  
	//Mat frameOri; //δ�������ԭʼ֡
	//video >> framePrePre;
	//video >> framePre;
	//cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);
	//cvtColor(framePre, framePre, CV_RGB2GRAY);
	//int save = 0;
	//while (true)
	//{
	//	video >> frameNow;
	//	frameNow.copyTo(frameOri);

	//	if (frameNow.empty() || waitKey(videoPause) == 27)
	//	{
	//		break;
	//	}
	//	cvtColor(frameNow, frameNow, CV_RGB2GRAY);
	//	Mat Det1;
	//	Mat Det2;
	//	absdiff(framePrePre, framePre, Det1);  //֡��1  
	//	absdiff(framePre, frameNow, Det2);     //֡��2  
	//	Mat meanMat, devMat;
	//	//������
	//	meanStdDev(Det1, meanMat, devMat);
	//	if (meanMat.at<double>(0, 0) >1 && devMat.at<double>(0, 0) >0.8)
	//		threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //����Ӧ��ֵ��  
	//	meanStdDev(Det2, meanMat, devMat);
	//	if (meanMat.at<double>(0, 0) >1 && devMat.at<double>(0, 0) >0.8)
	//		threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);
	//	Mat element = getStructuringElement(0, Size(3, 3));  //���ͺ�  
	//	dilate(Det1, Det1, element);    //����  
	//	dilate(Det2, Det2, element);
	//	bitwise_and(Det1, Det2, frameDet);
	//	framePrePre = framePre;
	//	framePre = frameNow;

	//	resize(frameOri, frameOri, Size(650, 550));
	//	resize(frameDet, frameDet, Size(650, 550));
	//	imshow(windowName, frameOri);
	//	imshow("Detection", frameDet);
	//}
	




	/******************************************���ڹ�����******************************************************/
	/*
	//����һ Ѱ��������
	Mat image1, image2;
	vector<Point2f> point1, point2, pointCopy;
	vector<uchar> status;
	vector<float> err;
	VideoCapture video(videoPath1);
	double fps = video.get(CV_CAP_PROP_FPS); //��ȡ��Ƶ֡��    
	double pauseTime = 1000 / fps; //���������м���     
	video >> image1;
	Mat image1Gray, image2Gray;
	cvtColor(image1, image1Gray, CV_RGB2GRAY);
	goodFeaturesToTrack(image1Gray, point1, 100, 0.01, 10, Mat());
	pointCopy = point1;
	for (int i = 0; i < point1.size(); i++)    //����������λ  
	{
		circle(image1, point1[i], 1, Scalar(0, 0, 255), 2);
	}
	namedWindow("�ǵ���������", 0);
	imshow("�ǵ���������", image1);
	while (true)
	{
		video >> image2;
		if (!image2.data || waitKey(pauseTime) == 27)  //ͼ��Ϊ�ջ�Esc�������˳�����    
		{
			break;
		}
		cvtColor(image2, image2Gray, CV_RGB2GRAY);
		calcOpticalFlowPyrLK(image1Gray, image2Gray, point1, point2, status, err, Size(20, 20), 3); //LK������       
		for (int i = 0; i < point2.size(); i++)
		{
			circle(image2, point2[i], 1, Scalar(0, 0, 255), 2);
			line(image2, pointCopy[i], point2[i], Scalar(255, 0, 0), 2);
		}
		imshow("�ǵ���������", image2);
		swap(point1, point2);
		image1Gray = image2Gray.clone();
	}
	*/

	//
	////������ 
	//VideoCapture cap;
	//cap.open(videoPath1);

	//if (!cap.isOpened()){
	//	std::cout << "��Ƶ��ȡʧ�ܣ�" << std::endl;
	//	return -1;
	//}

	//Mat  gray, prvGray, optFlow, absoluteFlow, img_for_show;
	//while (1){
	//	cap >> img;
	//	if (img.empty()) break;

	//	cvtColor(img, gray, CV_BGR2GRAY);
	//	if (prvGray.data){
	//		calcOpticalFlowFarneback(prvGray, gray, optFlow, 0.5, 3, 15, 3, 5, 1.2, 0); //ʹ�����Ĳ���  
	//		compute_absolute_mat(optFlow, absoluteFlow);
	//		normalize(absoluteFlow, img_for_show, 0, 255, NORM_MINMAX, CV_8UC1);
	//		imshow("opticalFlow", img_for_show);
	//		//imshow("opticalFlow", absoluteFlow);
	//		imshow("resource", img);
	//	}
	//	cv::swap(prvGray, gray);

	//	waitKey(1);
	//}

	

	/******************************���ڷ���*************************************/
	//ԭʼ����
	int n = 50;
	video.open(videoPath);
	if (!video.isOpened())
	{
		return -1;
	}
	namedWindow(windowName);
	createTrackbar(trackBarName, windowName, &trackbarValue, trackbarMax, TrackBarFunc);
	TrackBarFunc(0, 0);
	totalFrame = video.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ��֡�� 
	double videoFPS = video.get(CV_CAP_PROP_FPS);  //��ȡ֡��  
	double videoPause = 1000 / videoFPS;
	Mat frame, origin, result;

//	videoCap >> frame;
//	cvtColor(frame, frame, CV_RGB2GRAY);

	while (true)
	{
		video >> frame;
		if (frame.empty() || waitKey(videoPause) == 27)
		{
			break;
		}
		origin = frame.clone();//ԭʼ
		result = frame.clone();//���
		cvtColor(frame, frame, CV_RGB2GRAY);
		


		//ԭͼ�ϻ���
		for (int i = 0; i < frame.cols - frame.cols / n + 1; i += frame.cols / n){//����
			Point start = Point(i, 0);
			Point end = Point(i, frame.rows);
			line(result, start, end, Scalar(0, 255, 0));
		}
		for (int j = 0; j < frame.rows - frame.rows / n + 1; j += frame.rows / n){//����
			Point start = Point(0, j);
			Point end = Point(frame.cols, j);
			line(result, start, end, Scalar(0, 255, 0));
		}
		Mat Mean, Stddv;//��ֵ�ͷ���

		
		Mat mask = Mat::zeros(n, n, CV_8UC1);
		//���к���
//		for (int i = 0, im = 0; i < frame.cols - frame.cols / n + 1; i += frame.cols / n, im++){
//			for (int j = 0, jm = 0; j < frame.rows - frame.rows / n + 1; j += frame.rows / n, jm++){
		for (int j = 0, im = 0; j < frame.rows - frame.rows / n + 1 && im < n; j += frame.rows / n, im++){
			for (int i = 0, jm = 0; i < frame.cols - frame.cols / n + 1 && jm < n; i += frame.cols / n, jm++){
			
				Mat ROI = frame(Rect(i, j, frame.cols / n, frame.rows / n));

				//���������ֵ�ͷ���
				meanStdDev(ROI, Mean, Stddv);
				char meanStr[10];
				char stddvStr[10];
				string str = to_string(int(Stddv.at<double>(0, 0)));
				mask.at<uchar>(im, jm) = int(Stddv.at<double>(0, 0));
				threshold(mask, mask, 5, 255, THRESH_BINARY);
				putText(result, str, Point(i + frame.cols / (2 * n), j + frame.rows / (2 * n)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

			}
		}

		resize(origin, origin, Size(650, 550));
		//resize(result, result, Size(650, 550));
		imshow(windowName, result);
		//imshow("Detection", result);
		resize(mask, mask, Size(650, 550));
		imshow("Detection", mask);
	}
	








	//sobel����
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