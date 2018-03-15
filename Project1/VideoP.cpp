#include "VideoP.h"
//全局变量
int iPicNum = 0;//Set channel NO.
LONG nPort = -1;
HWND hWnd = NULL;
CRITICAL_SECTION g_cs_frameList;
std::list<Mat> g_frameList;
list<Mat> frameQueue;
HANDLE hEvent = NULL;
bool IsTracking = true;
int realframe_count = 0;

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
void VideoP::yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep){
	int col, row;
	unsigned int Y, U, V;
	int tmp;
	int idx;

	//printf("widthStep=%d.\n",widthStep);

	for (row = 0; row<height; row++)
	{
		idx = row * widthStep;
		int rowptr = row*width;

		for (col = 0; col<width; col++)
		{
			//int colhalf=col>>1;
			tmp = (row / 2)*(width / 2) + (col / 2);
			//         if((row==1)&&( col>=1400 &&col<=1600))
			//         { 
			//          printf("col=%d,row=%d,width=%d,tmp=%d.\n",col,row,width,tmp);
			//          printf("row*width+col=%d,width*height+width*height/4+tmp=%d,width*height+tmp=%d.\n",row*width+col,width*height+width*height/4+tmp,width*height+tmp);
			//         } 
			Y = (unsigned int)inYv12[row*width + col];
			U = (unsigned int)inYv12[width*height + width*height / 4 + tmp];
			V = (unsigned int)inYv12[width*height + tmp];
			//         if ((col==200))
			//         { 
			//         printf("col=%d,row=%d,width=%d,tmp=%d.\n",col,row,width,tmp);
			//         printf("width*height+width*height/4+tmp=%d.\n",width*height+width*height/4+tmp);
			//         return ;
			//         }
			if ((idx + col * 3 + 2)> (1200 * widthStep))
			{
				//printf("row * widthStep=%d,idx+col*3+2=%d.\n",1200 * widthStep,idx+col*3+2);
			}
			outYuv[idx + col * 3] = Y;
			outYuv[idx + col * 3 + 1] = U;
			outYuv[idx + col * 3 + 2] = V;
		}
	}
	//printf("col=%d,row=%d.\n",col,row);
}
void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2){
	long lFrameType = pFrameInfo->nType;

	if (lFrameType == T_YV12)
	{
#if USECOLOR
		//int start = clock();
		static IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//得到图像的Y分量  
		yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//得到全部RGB图像
		static IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);
		cvCvtColor(pImgYCrCb, pImg, CV_YCrCb2RGB);
		//int end = clock();
#else
		static IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 1);
		memcpy(pImg->imageData, pBuf, pFrameInfo->nWidth*pFrameInfo->nHeight);
#endif
		//printf("%d\n",end-start);

		Mat frametemp(pImg), frame;

		//frametemp.copyTo(frame);
		//      cvShowImage("IPCamera",pImg);
		//      cvWaitKey(1);
		EnterCriticalSection(&g_cs_frameList);
		g_frameList.push_back(frametemp);
		LeaveCriticalSection(&g_cs_frameList);

#if USECOLOR
		//      cvReleaseImage(&pImgYCrCb);
		//      cvReleaseImage(&pImg);
#else
		/*cvReleaseImage(&pImg);*/
#endif
		//此时是YV12格式的视频数据，保存在pBuf中，可以fwrite(pBuf,nSize,1,Videofile);
		//fwrite(pBuf,nSize,1,fp);
	}
	/***************
	else if (lFrameType ==T_AUDIO16)
	{
	//此时是音频数据，数据保存在pBuf中，可以fwrite(pBuf,nSize,1,Audiofile);

	}
	else
	{

	}
	*******************/

}
void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser){
	DWORD dRet;
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
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

			//设置解码回调函数 解码且显示
			//if (!PlayM4_SetDecCallBackEx(nPort,DecCBFun,NULL,NULL))
			//{
			//  dRet=PlayM4_GetLastError(nPort);
			//  break;
			//}

			//打开视频解码
			if (!PlayM4_Play(nPort, hWnd))
			{
				dRet = PlayM4_GetLastError(nPort);
				break;
			}

			//打开音频解码, 需要码流是复合流
			//          if (!PlayM4_PlaySound(nPort))
			//          {
			//              dRet=PlayM4_GetLastError(nPort);
			//              break;
			//          }       
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
void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser){
	char tempbuf[256] = { 0 };
	switch (dwType)
	{
	case EXCEPTION_RECONNECT:    //预览时重连
		printf("----------reconnect--------%d\n", time(NULL));
		break;
		default:
		break;
	}
}
unsigned CALLBACK readCamera(void *param){
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
		return -1;
	}

	//---------------------------------------
	//设置异常消息回调函数
	NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);


	//cvNamedWindow("IPCamera");
	//---------------------------------------
	//启动预览并设置回调数据流 
	NET_DVR_CLIENTINFO ClientInfo;
	ClientInfo.lChannel = 1;        //Channel number 设备通道号
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

	//cvWaitKey(0);
	Sleep(-1);

	//fclose(fp);
	//---------------------------------------
	//关闭预览
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

DWORD WINAPI dealFun(LPVOID lpParamter){
	while (1)
	{
		if (!frameQueue.empty()){
			WaitForSingleObject(hEvent, INFINITE);
			//src = (Mat)(*(frameQueue.begin()));//frameQueue.front();  
			Mat src_YCrCb = (Mat)(*(frameQueue.begin()));
			Mat src;
			cvtColor(src_YCrCb, src, CV_YCrCb2BGR);
			frameQueue.pop_front();
			SetEvent(hEvent);

			//your code............  


		}

	}
	return 0;
}
void  VideoP::showVideo(){
	HANDLE hThread;
	unsigned threadID;
	Mat frame1;

	InitializeCriticalSection(&g_cs_frameList);
	hThread = (HANDLE)_beginthreadex(NULL, 0, &readCamera, NULL, 0, &threadID);
	
		EnterCriticalSection(&g_cs_frameList);
	if (g_frameList.size())
	{
		list<Mat>::iterator it;
		it = g_frameList.end();
		it--;
		Mat dbgframe = (*(it));
		imshow("frame from camera",dbgframe);
		//dbgframe.copyTo(frame1);
		//dbgframe.release();
		/*(*g_frameList.begin()).copyTo(frame[i]);*/
		frame1 = dbgframe;
		g_frameList.pop_front();
	}
	g_frameList.clear(); // 丢掉旧的帧
	LeaveCriticalSection(&g_cs_frameList);

}