#include "ImageP.h"
#include "HogSVM.h"
#include "VideoP.h"

int main(){

	ImageP Processor;
	string PicPath = "F:\\ú̿ʶ��\\���ʹ����\\6.png";
	
	//Mat img = imread(PicPath);
	//img = cvtColor(img, img , )
	Processor.PiexLocation_Show(PicPath);

	getchar();
	return 0;
}