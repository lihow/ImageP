#include "ImageP.h"
#include "HogSVM.h"
#include "VideoP.h"

int main(){

	ImageP Processor;
	string PicPath = "C:\\Users\\Lenovo\\Desktop\\R1.jpg";
	string PicPath1 = "F:\\±œ…Ë\\A±œ…ËÕº∆¨\\class\\10\\1.jpg";
	
	//Processor.MoneyROI(PicPath1);

	//Processor.GetContoursPic(PicPath, "C:\\Users\\Lenovo\\Desktop\\result.jpg");
	//Processor.FindDiff(PicPath,true);
	//Processor.LineFind(PicPath);
	//Processor.BackgroundTransfer(PicPath);
	//Processor.Blur(Processor.AddSaltNoise(PicPath,3000,false));

	HogSVM hogsvm;
	string TrainPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\MoneyR\\TrainTxt.txt";
	string LabelPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\MoneyR\\LabelTxt.txt";
	string TestPath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\MoneyR\\TestTxt.txt";
	string labelnamepath = "E:\\imageP\\c++\\ImageP\\Project1\\data\\MoneyR\\Label2name.txt";
	//hogsvm.trian(TrainPath, LabelPath, "MoneyRSVM.xml");
	hogsvm.predict(TestPath, labelnamepath, "MoneyRSVM.xml");

	//Processor.LBP(PicPath);

	VideoP videop;
	string VideoPath = "C:\\Users\\Lenovo\\Desktop\\Mei\\Mei-part.avi";
	//videop.LocalCamera();
	//videop.VideoBackgroundSubtractor(VideoPath);
	//videop.showVideo();
	//videop.HKshowVideo();

	string refCoalPath = "C:\\Users\\Lenovo\\Desktop\\ref.jpg";
	string curCoalPath = "C:\\Users\\Lenovo\\Desktop\\coal.jpg";
	//Processor.CountWdith(refCoalPath, curCoalPath);

	getchar();
	return 0;
}