#ifndef VIDEOP_H
#define VIDEOP_H

#include "ImageP.h"
//һ��ͷ�ļ�
#include <windows.h>
#include <time.h>
#include <thread>
#include <process.h> 
//SDK ͷ�ļ�
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
	//���ú�������SDK
	void HKshowVideo();
	//����VLC��ȡ����ͷ
	void VLCshowVideo();
	//������Ƶ
	void PlayVideo(const string VideoPath);
private:
	//������Ƶ����
	//void on_Trackbar(int, void*);
	ImageP ImgP;
	void ShowHelpText();
	void ImageText(Mat* img, const char* text, int x, int y);

};





#endif