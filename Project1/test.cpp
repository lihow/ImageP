#include "ImageP.h"
#include "HogSVM.h"
#include "VideoP.h"

int main(){

	ImageP Processor;
	string PicPath = "C:\\Users\\Lenovo\\Desktop\\mmexport1520761587181.jpg";
	//Processor.FindDiff(PicPath);
	//Processor.LineFind(PicPath);
	
	//Processor.Blur(Processor.AddSaltNoise(PicPath,3000,false));

	HogSVM hogsvm;
	string TrainPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\svm_images\\TrainTxt.txt";
	string LabelPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\svm_images\\LabelTxt.txt";
	string TestPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\svm_images\\TestTxt.txt";
	string labelnamepath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\svm_images\\Label2name.txt";
	//hogsvm.trian(TrainPath, LabelPath);
	//hogsvm.predict(TestPath, labelnamepath);

	//Processor.LBP(PicPath);

	VideoP videop;
	string VideoPath = "C:\\Users\\Lenovo\\Desktop\\Mei\\Mei-part.avi";
	//videop.LocalCamera();
	//videop.VideoBackgroundSubtractor(VideoPath);
	//videop.showVideo();
	videop.LocalCamera();

	string refCoalPath = "C:\\Users\\Lenovo\\Desktop\\ref.jpg";
	string curCoalPath = "C:\\Users\\Lenovo\\Desktop\\coal.jpg";
	//Processor.CountWdith(refCoalPath, curCoalPath);

	getchar();
	return 0;
}