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
/**********************************************************************************

以下内容为调用海康威视SDK操作

***********************************************************************************/

#include <windows.h>
#include <list>
#include <time.h>

#include <HCNetSDK.h>
#include <plaympeg4.h>
#include <process.h>

#include "PeopleDetect.h"

#define USECOLOR 1
#define BUFFER_SIZE 15


//缓冲区队列锁
CRITICAL_SECTION g_cs_frameList;
CRITICAL_SECTION g_cs_peoples;
//缓冲区队�?
list<Mat> g_frameList;
//行人检测结�?(矩形�?)
vector<Rect> peoples;

//行人检测器
PeoDetect mPD;

//SDK相关
int iPicNum = 0;//Set channel NO.
LONG nPort = -1;
HWND hWnd = NULL;

//解码回调数据
void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep);
void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2);
//实时回调�?
void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser);
void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser);
//读取摄像头图像线�?
unsigned __stdcall readCamera(void *param);
//行人处理线程
unsigned __stdcall process_people(void *param);

int VideoP::showVideo()
{
	HANDLE hGetFrame, hProcess_people;
	unsigned tidGetFrame, tidProcess_people;
	Mat frame;
	InitializeCriticalSection(&g_cs_frameList);
	InitializeCriticalSection(&g_cs_peoples);
	hGetFrame = (HANDLE)_beginthreadex(NULL, 0, &readCamera, NULL, 0, &tidGetFrame);
	//hProcess_people = (HANDLE)_beginthreadex(NULL, 0, &process_people, NULL, 0, &tidProcess_people);
	Mat dbgframe;
	vector<Rect> peo_temp;
	Mat frame1;
	while (1){
		//if (g_frameList.size())
		//{
		//	EnterCriticalSection(&g_cs_frameList);
		//	dbgframe = g_frameList.front();
		//	g_frameList.pop_front();
		//	LeaveCriticalSection(&g_cs_frameList);

		//	if (dbgframe.cols * dbgframe.rows != 0){
		//		//如果检测到人脸，则进行绘制
		//		if (peoples.size()){
		//			EnterCriticalSection(&g_cs_peoples);
		//			peo_temp = peoples;
		//			LeaveCriticalSection(&g_cs_peoples);
		//			for (Rect r : peo_temp)
		//				rectangle(dbgframe, r, Scalar(0, 255, 0), 2, 8, 0);
		//		}
		//	}
		//	imshow("Result", dbgframe);
		//	cv::waitKey(1);
		//}
	
		if (g_frameList.size())
		{
			list<Mat>::iterator it;
			it = g_frameList.end();
			it--;
			Mat dbgframe = (*(it));
			resize(dbgframe, dbgframe, Size(500,500));
			imshow("frame from camera",dbgframe);
			cv::waitKey(1);
			//dbgframe.copyTo(frame1);
			//dbgframe.release();
			//(*g_frameList.begin()).copyTo(frame[i]);
			frame1 = dbgframe;
			g_frameList.pop_front();
		}
	}
	g_frameList.clear(); // 丢掉旧的�?
	ExitThread(tidGetFrame);
	ExitThread(tidProcess_people);
	system("pause");
	return 0;
}

void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep)
{
	int col, row;
	unsigned int Y, U, V;
	int tmp;
	int idx;
	for (row = 0; row<height; row++)
	{
		idx = row * widthStep;
		int rowptr = row*width;
		for (col = 0; col<width; col++)
		{
			tmp = (row / 2)*(width / 2) + (col / 2);
			Y = (unsigned int)inYv12[row*width + col];
			U = (unsigned int)inYv12[width*height + width*height / 4 + tmp];
			V = (unsigned int)inYv12[width*height + tmp];
			outYuv[idx + col * 3] = Y;
			outYuv[idx + col * 3 + 1] = U;
			outYuv[idx + col * 3 + 2] = V;
		}
	}
}

//解码回调 视频为YUV数据(YV12)，音频为PCM数据
void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2)
{
	long lFrameType = pFrameInfo->nType;

	if (lFrameType == T_YV12)
	{
#if USECOLOR
		static IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//得到图像的Y分量
		yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//得到全部RGB图像
		static IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);
		cvCvtColor(pImgYCrCb, pImg, CV_YCrCb2RGB);
#else
		static IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 1);
		memcpy(pImg->imageData, pBuf, pFrameInfo->nWidth*pFrameInfo->nHeight);
#endif
		Mat frametemp(pImg);
		//resize(frametemp, frametemp, Size(640, 480));
		//加锁并将图像压入缓冲区队�?
		EnterCriticalSection(&g_cs_frameList);
		//队列最大长度限�?
		if (g_frameList.size() > BUFFER_SIZE)
			g_frameList.pop_front();
		g_frameList.push_back(frametemp);
		LeaveCriticalSection(&g_cs_frameList);
#if USECOLOR
#else
		cvReleaseImage(&pImg);
#endif
	}
}

///实时流回�?
void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统�?
		if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道�?
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 1024 * 1024))
			{
				dRet = PlayM4_GetLastError(nPort);
				break;
			}
			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort, DecCBFun))
			{
				dRet = PlayM4_GetLastError(nPort);
				break;
			}
			//打开视频解码
			if (!PlayM4_Play(nPort, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort != -1)
		{
			BOOL inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
				OutputDebugString(L"PlayM4_InputData failed \n");
			}
		}
		break;
	}
}

void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
{
	char tempbuf[256] = { 0 };
	switch (dwType)
	{
	case EXCEPTION_RECONNECT:    //预览时重�?
		printf("----------reconnect--------%d\n", time(NULL));
		break;
	default:
		break;
	}
}

unsigned __stdcall readCamera(void *param)
{
	//---------------------------------------
	// 初始�?
	NET_DVR_Init();
	//设置连接时间与重连时�?
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);
	// 注册设备
	LONG lUserID;
	NET_DVR_DEVICEINFO_V30 struDeviceInfo;
	lUserID = NET_DVR_Login_V30("192.168.1.64", 8000, "admin", "admin888", &struDeviceInfo);
	if (lUserID < 0)
	{
		printf("Login error, %d\n", NET_DVR_GetLastError());
		NET_DVR_Cleanup();
		return -1;
	}

	//---------------------------------------
	//设置异常消息回调函数
	NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);

	//cvNamedWindow("IPCamera");
	//---------------------------------------
	//启动预览并设置回调数据流
	NET_DVR_CLIENTINFO ClientInfo;
	ClientInfo.lChannel = 1;        //Channel number 设备通道�?
	ClientInfo.hPlayWnd = NULL;     //窗口为空，设备SDK不解码只取流
	ClientInfo.lLinkMode = 0;       //Main Stream
	ClientInfo.sMultiCastIP = NULL;

	LONG lRealPlayHandle;
	lRealPlayHandle = NET_DVR_RealPlay_V30(lUserID, &ClientInfo, fRealDataCallBack, NULL, TRUE);
	if (lRealPlayHandle<0)
	{
		printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
		return 0;
	}
	Sleep(-1);
	if (!NET_DVR_StopRealPlay(lRealPlayHandle))
	{
		printf("NET_DVR_StopRealPlay error! Error number: %d\n", NET_DVR_GetLastError());
		return 0;
	}
	//注销用户
	NET_DVR_Logout(lUserID);
	NET_DVR_Cleanup();
	return 0;
}

unsigned __stdcall process_people(void *param)
{
	Mat src, colorful, mask;
	vector <Rect> mPeoples;
	while (1){
		if (g_frameList.size()){
			EnterCriticalSection(&g_cs_frameList);
			src = g_frameList.front();
			LeaveCriticalSection(&g_cs_frameList);
			if (!src.empty())
			{
				if (src.channels() == 3){
					colorful = src.clone();
					cvtColor(src, src, COLOR_BGR2GRAY);
				}
				//mPeoples 存储人脸矩形框序�?
				mPeoples = mPD.detectPeople(src);
				peoples = mPeoples;
			}
		}
	}
	return 0;
}