#ifndef HOGSVM_H
#define HOGSVM_H

#include"MachineL.h"

/*
����hog+svm�ṩѵ����Ԥ��
ǰ��׼����
TrainTxt.txt����¼ѵ���ļ���ַ ����ΪE:\airplanes\0_airplanes_1.jpg
LabelTxt.txt����¼ѵ���ļ�������ǩ ����Ϊ0��0��0��1��1��1...��Ӧ�ڲ�ͬ�����
Label2name.txt��������ƣ���Ӧ��LabelTxt.txt�еı�ǩ
TestTxt.txt�������ļ��ĵ�ַ
��Ҫ������
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
	Size imageSize = Size(64, 128);
	void coumputeHog(const Mat & src, vector<float> & descriptors);
	ImageP imagep;
};

#endif