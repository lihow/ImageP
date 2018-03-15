#ifndef VIDEOP_H
#define VIDEOP_H

#include "ImageP.h"
//SDK 头文件
#include "HCNetSDK.h"  
#include "plaympeg4.h"
//一般头文件
#include <windows.h>
#include <time.h>
#include <thread>
#include <process.h> 

class VideoP{
public:
	void LocalCamera();
	void VideoBackgroundSubtractor(const string VideoPath);
	//调用海康威视SDK
	void showVideo();
private:
	//调用海康威视SDK
	void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep);
	
	

};

//海康威视SDK 定义为全局变量和全局函数

void  CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2);
void  CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser);
void  CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser);
//DWORD WINAPI dealFun(LPVOID lpParamter);
unsigned CALLBACK readCamera(void *param);




#endif