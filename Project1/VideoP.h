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
	
	

};





#endif