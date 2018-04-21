#include"HogSVM.h"

HogSVM::HogSVM(){
	mySVM = new CvSVM();
	params = CvSVMParams();
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-10);
}
HogSVM::~HogSVM(){
	delete mySVM;
}
void HogSVM::coumputeHog(const Mat & src, vector<float> & descriptors){
	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));
}
/*
@function:ѵ������
@param TrainPath:��¼ѵ���ļ�·�����ı���ַ
@param LabelPath:ѵ���ļ������ֱ�ǩ
@param XmalPath:ѵ��ģ�͵�ַ����λ��
*/
void HogSVM::trian(const string TrainPath, const string LabelPath, const string XmlPath){
	ifstream inLabels(LabelPath), inImages(TrainPath);
	
	string imageName;
	string imageLabel;
	vector<Mat> vecImages;
	vector<int> vecLabels;
	vector<float> vecDescriptors;

	cout << "��ȡͼƬ�ͱ�ǩ..." << endl;
	//while ((inImages >> imageName) && (inLabels >> imageLabel))
	while (getline(inImages, imageName) && getline(inLabels, imageLabel))
	{
		//ԭʼ����
		//Mat src = imread(imageName, 0);

		/*********************************************����ֽ��ʶ��****************************************/
		cout << imageName << endl;
		Mat src = imagep.MoneyROI(imageName, false);

		resize(src, src, imageSize);
		vecImages.push_back(src);
		vecLabels.push_back(atoi(imageLabel.c_str()));
	}
	inLabels.close();
	inImages.close();

	Mat dataDescriptors;
	Mat dataResponse = (Mat)vecLabels;
	for (size_t i = 0; i < vecImages.size(); i++)
	{
		Mat src = vecImages[i];
		Mat tempRow;
		coumputeHog(src, vecDescriptors);
		if (i == 0)
		{
			dataDescriptors = Mat::zeros(vecImages.size(), vecDescriptors.size(), CV_32FC1);
		}
		tempRow = ((Mat)vecDescriptors).t();
		tempRow.row(0).copyTo(dataDescriptors.row(i));
	}
	cout << "ѵ����..." << endl;
	mySVM->train(dataDescriptors, dataResponse, Mat(), Mat(), params);
	mySVM->save(XmlPath.c_str());
	cout << "ѵ�����!" << endl << "���������" << XmlPath << endl;
}
/*
@function:Ԥ�⺯��
@param TestPath:��ҪԤ����ļ���ַ��¼�ı�
@param LabelnamePath:����ı� ��Ӧ��ѵ��ʱ�����ֱ�ǩ
@param XmlPath:ѵ��ģ�͵�ַ
*/
void HogSVM::predict(const string TestPath, const string LabelnamePath, const string XmlPath){
	ifstream inTestimage(TestPath), inLabelname(LabelnamePath);
	string labelname;
	vector<string> names;
	while (getline(inLabelname, labelname)){
		names.push_back(labelname);
	}
	
	mySVM->load(XmlPath.c_str());

	string testPath;
	//while (inTestimage >> testPath)
	while (getline(inTestimage, testPath))
	{
		//ԭʼ����
		//Mat test = imread(testPath, 0);

		/************************************����ֽ��ʶ���******************************************/
		Mat test = imagep.MoneyROI(testPath, false);

		resize(test, test, imageSize);
		vector<float> imageDescriptor;
		coumputeHog(test, imageDescriptor);
		Mat testDescriptor = Mat::zeros(1, imageDescriptor.size(), CV_32FC1);
		for (size_t i = 0; i < imageDescriptor.size(); i++)
		{
			testDescriptor.at<float>(0, i) = imageDescriptor[i];
		}
		float  label = mySVM->predict(testDescriptor, false);
		cout << names[label] << endl;
		//cout << label << endl;
		imshow("test image", test);
		waitKey(0);
	}
	inTestimage.close();
}
