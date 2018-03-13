#ifndef HOGSVM_H
#define HOGSVM_H

#include"MachineL.h"

/*
利用hog+svm提供训练和预测
前期准备：
TrainTxt.txt：记录训练文件地址 内容为E:\airplanes\0_airplanes_1.jpg
LabelTxt.txt：记录训练文件的类别标签 内容为0，0，0，1，1，1...对应于不同的类别
Label2name.txt：类别名称，对应于LabelTxt.txt中的标签
TestTxt.txt：测试文件的地址
主要函数：
void trian(const string TrainPath, const string LabelPath, const string XmlPath = "HogSVM.xml");
void predict(const string TestPath, const string LabelnamePath, const string XmlPath = "HogSVM.xml");
*/

class HogSVM{
public:
	HogSVM();
	~HogSVM();
	void trian(const string TrainPath, const string LabelPath, const string XmlPath = "HogSVM.xml");
	void predict(const string TestPath, const string LabelnamePath, const string XmlPath = "HogSVM.xml");
private:
	//string TrainPath;
	//string LabelPath;
	//string TestPath;
	CvSVM *mySVM;
	CvSVMParams params;
	Size imageSize = Size(64, 64);
	void coumputeHog(const Mat & src, vector<float> & descriptors);
};

#endif