#include "VideoP.h"


/*
@function:读取本地摄像头和网络摄像头
*/
void VideoP::LocalCamera(){
	VideoCapture cap;
	string rtsp_addr("rtsp://admin:admin888@192.168.1.64:554/mpeg4/ch1/main/av_stream");
	//cap.open(0);
	cap.open(rtsp_addr);
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

#define USECOLOR 0
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
//补
bool IsTracking = true;
 HANDLE hThread1;  
 HANDLE hThread2;
 HANDLE hEvent;
 int realframe_count = 0;
 list<Mat>frameQueue;

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
	//TRACE("DecCBFun 函数被调用\n");  
	long lFrameType = pFrameInfo->nType;
	//TRACE(" lFrameType: %ld\n", lFrameType);  

	if (lFrameType == T_YV12)
	{

#if USECOLOR
		Mat pImg(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
		Mat pImg_YUV(pFrameInfo->nHeight + pFrameInfo->nHeight / 2, pFrameInfo->nWidth, CV_8UC1, pBuf);
		Mat pImg_YCrCb(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
		cvtColor(pImg_YUV, pImg, CV_YUV2BGR_YV12);  
		cvtColor(pImg,pImg_YCrCb,CV_BGR2YCrCb); 
#else
		Mat pImg(pFrameInfo->nHeight + pFrameInfo->nHeight / 2, pFrameInfo->nWidth, CV_8UC1, pBuf);
#endif
		//  Sleep(-1);  
		resize(pImg, pImg, Size(500, 500));
		imshow("IPCamera", pImg);
		
		waitKey(1);
		//waitKey(1);  
		//IplImage *pImg1 = &IplImage(pImg);  
		if (!IsTracking){
			hEvent = CreateEvent(NULL, false, true, NULL);
			//InitializeCriticalSection(&cs_frameQueue);  
		}

		//HANDLE hThread = CreateThread(NULL, 0, dealFun, NULL, 0, NULL);  
		//CloseHandle(hThread);   
		//图片存储  
		//*--------回调函数当做存储视频帧线程-----------  
		//ResetEvent(hEvent);  

		//EnterCriticalSection(&cs_frameQueue);  
		realframe_count++;
		//TRACE("实时帧数: %d\n",realframe_count);  
		if (0 == realframe_count % 10)
		{
			WaitForSingleObject(hEvent, INFINITE);

			frameQueue.push_back(pImg);
			if (!IsTracking){
				frameQueue.clear();
			}

			SetEvent(hEvent);
		}

		//LeaveCriticalSection(&cs_frameQueue);  
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


void VideoP::HKshowVideo()
{
	//---------------------------------------    
	// 初始化    
	NET_DVR_Init();
	//设置连接时间与重连时间    
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);

	//---------------------------------------    
	// 获取控制台窗口句柄    
	//HMODULE hKernel32 = GetModuleHandle((LPCWSTR)"kernel32");    
	//GetConsoleWindow = (PROCGETCONSOLEWINDOW)GetProcAddress(hKernel32,"GetConsoleWindow");    

	//---------------------------------------    
	// 注册设备    
	LONG lUserID;
	NET_DVR_DEVICEINFO_V30 struDeviceInfo;
	lUserID = NET_DVR_Login_V30("192.168.1.64", 8000, "admin", "admin888", &struDeviceInfo);
	if (lUserID < 0)
	{
		printf("Login error, %d\n", NET_DVR_GetLastError());
		NET_DVR_Cleanup();
		return;
	}

	//---------------------------------------    
	//设置异常消息回调函数    
	NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);


	//NET_DVR_RealPlay_V30参数设置
	NET_DVR_CLIENTINFO ClientInfo;
	ClientInfo.hPlayWnd     = NULL;//改为“= GetDlgItem(IDC_STATIC_PLAY)->m_hWnd”
	ClientInfo.lChannel     = 1;
	ClientInfo.lLinkMode    = 0;
	ClientInfo.sMultiCastIP = NULL;
	//TRACE("Channel number:%d\n",ClientInfo.lChannel);

	//NET_DVR_RealPlay_V40参数设置  
	//NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
	//struPlayInfo.hPlayWnd = NULL;    //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空  
	//struPlayInfo.lChannel = 1;       //预览通道号  
	//struPlayInfo.dwStreamType = 0;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推  
	//struPlayInfo.dwLinkMode = 0;       //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP  




	LONG lRealPlayHandle;
	lRealPlayHandle = NET_DVR_RealPlay_V30(lUserID, &ClientInfo, fRealDataCallBack, NULL, false);
	//lRealPlayHandle = NET_DVR_RealPlay_V40(lLoginID, &struPlayInfo, fRealDataCallBack, NULL);

	if (lRealPlayHandle<0)
	{
		printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
		return;
	}

	//cvWaitKey(0);    
	Sleep(-1);

	//fclose(fp);    
	//---------------------------------------    
	//关闭预览    
	if (!NET_DVR_StopRealPlay(lRealPlayHandle))
	{
		printf("NET_DVR_StopRealPlay error! Error number: %d\n", NET_DVR_GetLastError());
		return;
	}
	//注销用户    
	NET_DVR_Logout(lUserID);
	NET_DVR_Cleanup();

	return;
}

/*******************************************************************************************

以下为通过VLC解码，调用摄像头

********************************************************************************************/
int VIDEO_WIDTH = 1024;
int VIDEO_HEIGHT = 578;
static char * videobuf = 0;
string Vlc_Vertion = "";
void *lock(void *data, void**p_pixels)
{
	*p_pixels = videobuf;
	return NULL;
}
void display(void *data, void *id)
{
	IplImage *img = cvCreateImage(cvSize(VIDEO_WIDTH, VIDEO_HEIGHT), IPL_DEPTH_8U, 4);
	img->imageData = videobuf;
	cvShowImage(libvlc_get_version(), img);
	cvWaitKey(10);
	cvReleaseImage(&img);
}
void unlock(void *data, void *id, void *const *p_pixels)
{
	(void)data;
	assert(id == NULL);
}

void VideoP::VLCshowVideo()
{
	cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
	libvlc_media_t* media = NULL;
	libvlc_media_player_t* mediaPlayer = NULL;
	char const* vlc_args[] =
	{
		"-I",
		"dummy",
		"--ignore-config",
	};
	Vlc_Vertion = libvlc_get_version();
	videobuf = (char*)malloc((VIDEO_WIDTH * VIDEO_HEIGHT) << 2);
	memset(videobuf, 0, (VIDEO_WIDTH * VIDEO_HEIGHT) << 2);

	libvlc_instance_t* instance = libvlc_new(sizeof(vlc_args) / sizeof(vlc_args[0]), vlc_args);

	media = libvlc_media_new_location(instance, "rtsp://admin:admin888@192.168.1.64:554");
	//media = libvlc_media_new_location(instance, "file:///C:\\Users\\Lenovo\\Desktop\\Mei\\Mei-part.avi");
	mediaPlayer = libvlc_media_player_new_from_media(media);
	libvlc_media_release(media);

	//libvlc_media_player_set_media(mediaPlayer, media);  
	libvlc_video_set_callbacks(mediaPlayer, lock, unlock, display, NULL);
	libvlc_video_set_format(mediaPlayer, "RV32", VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_WIDTH << 2);
	libvlc_media_player_play(mediaPlayer);


}