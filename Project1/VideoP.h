#ifndef VIDEOP_H
#define VIDEOP_H

#include "ImageP.h"
//一般头文件
#include <windows.h>
#include <time.h>
#include <thread>
#include <process.h> 
//SDK 头文件
//HK
#include "HCNetSDK.h"  
#include "plaympeg4.h"
//VLC
#include "vlc/vlc.h"
#include "vlc/libvlc_media.h"
#include "vlc/libvlc_media_player.h"
class VideoP{
public:
	void LocalCamera();
	void VideoBackgroundSubtractor(const string VideoPath);
	//调用海康威视SDK
	void HKshowVideo();
	//调用VLC读取摄像头
	void VLCshowVideo();
	//播放视频
	void PlayVideo(const string VideoPath);
private:
	//播放视频调用
	//void on_Trackbar(int, void*);
	ImageP ImgP;
	void ShowHelpText();
	void ImageText(Mat* img, const char* text, int x, int y);

};





#endif