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
	
	

};





#endif