#ifndef VIDEOP_H
#define VIDEOP_H

#include "ImageP.h"
//SDK ͷ�ļ�
#include "HCNetSDK.h"  
#include "plaympeg4.h"
//һ��ͷ�ļ�
#include <windows.h>
#include <time.h>
#include <thread>
#include <process.h> 

class VideoP{
public:
	void LocalCamera();
	void VideoBackgroundSubtractor(const string VideoPath);
	//���ú�������SDK
	void showVideo();
private:
	//���ú�������SDK
	void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep);
	
	

};

//��������SDK ����Ϊȫ�ֱ�����ȫ�ֺ���

void  CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2);
void  CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser);
void  CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser);
//DWORD WINAPI dealFun(LPVOID lpParamter);
unsigned CALLBACK readCamera(void *param);




#endif