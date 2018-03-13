#include"ImageP.h"

/*
����΢��С���������Ҳ�
���룺
PicPath:�����ļ�·��
outpath:������·��
show:�Ƿ�չʾ�м����н��ͼ
�����
���ͼ
*/
Mat ImageP::FindDiff(const string PicPath, const string OutPath, bool show){
	cv::Mat image;
	image = cv::imread(PicPath);

	cv::Mat ROI_1 = image(cv::Rect(210, 170, 829, 826));
	cv::Mat ROI_2 = image(cv::Rect(210, 1031, 829, 826));

	cv::Mat result = ROI_1 - ROI_2;
	cv::threshold(result, result, 10, 255, cv::THRESH_BINARY_INV);
	result = 0.8*result + 0.2*ROI_1;
	cv::imwrite(OutPath, result);
	cout << "��������ڣ�" << OutPath << endl;

	if (show){
		cv::namedWindow("ROI1");
		cv::imshow("ROI1", ROI_1);
		cv::namedWindow("ROI2");
		cv::imshow("ROI2", ROI_2);
		cv::imshow("Result Image", result);
		cv::waitKey();
	}
	return result;
}
/*
����sift������ƥ������ͼƬ
���룺
PicPath_1:����ͼƬ1λ��
PicPath_2:����ͼƬ2λ��
OutPath:���λ��
show:�Ƿ�չʾ�м����
�����
������ͼ
*/
Mat ImageP:: SiftMatch(const string PicPath_1, const string PicPath_2, const string OutPath, bool show){
	Mat img1 = imread(PicPath_1);
	Mat img2 = imread(PicPath_2);
	if (img1.empty()){
		cout<<"Cannot load image "<< PicPath_1<<endl;
	}
	if (img2.empty()){
		cout << "Cannot load image " << PicPath_2 << endl;
	}
	if (show){
		imshow("image1 before", img1);
		waitKey(10);
		imshow("image2 before", img2);
		waitKey(10);
	}
	//sift��������
	SiftFeatureDetector siftdtc;
	vector<KeyPoint> kp1, kp2;
	siftdtc.detect(img1, kp1);
	Mat outimg1;
	drawKeypoints(img1, kp1, outimg1);
	siftdtc.detect(img2, kp2);
	Mat outimg2;
	drawKeypoints(img2, kp2, outimg2);
	//����������
	SiftDescriptorExtractor ext;
	Mat descp1, descp2;
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	Mat img_matches;
	ext.compute(img1, kp1, descp1);
	ext.compute(img2, kp2, descp2);
	matcher.match(descp1, descp2, matches);

	drawMatches(img1, kp1, img2, kp2, matches, img_matches);
	cv::imwrite(OutPath, img_matches);
	cout << "��������ڣ�" << OutPath << endl;
	
	if (show){
		imshow("desc", descp1);
		imshow("matches", img_matches);
		waitKey();
	}

	return img_matches;
	
}
/*
@function: surf��������ȡ
@param PicPath:ͼƬ����·��
@param show:�Ƿ�չʾ�м���
@return ��ͼ���surf����ͼ
*/
Mat ImageP::SurfFea(const string PicPath, bool show){
	Mat image = imread(PicPath);
	if (image.empty()){
		cout << "Cannot load image:" << PicPath << endl;
	}
	if (show){
		imshow("image before", image);
		waitKey(10);
	}
	Mat outimage;
	vector<KeyPoint> keypoints;
	SurfFeatureDetector surf(3000);
	surf.detect(image, keypoints);
	drawKeypoints(image, keypoints, outimage, Scalar(120,0,120), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	if (show){
		imshow("SURF", outimage);
		waitKey();
	}
	return outimage;
}
/*
@function: ����ͼ���е�hog�������������
@param PicPath:����ͼƬ·��
@param show:�Ƿ�չʾ�м���
@return ����Ľ��ͼ
*/
Mat ImageP::HogPeople(const string PicPath, bool show){
	Mat image = imread(PicPath);
	if (image.empty()){
		cout << "Cannot load image: " << PicPath << endl;
	}
	if (show){
		imshow("image before", image);
		waitKey(10);
	}
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	vector<Rect>regions;
	hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);
	for (size_t i = 0; i < regions.size(); i++){
		rectangle(image, regions[i], Scalar(0, 0, 255), 2);
	}
	if (show){
		imshow("Pople image", image);
		waitKey();
	}
	return image;
}
/*
@function: ���ȶ�ͼ����и�˹ů����˲����н��봦���ٲ���Laplace���ӽ��б�Ե��⣬�Ϳ���������Ӷ���������ɢ���Robust, ��һ��������Laplacian of Gaussian(LOG)���Ӿ͵�����
			������Ҳ�����˸�˹�任��Sobel, Canny�任
@PicPath: ͼƬ·��
@show:�Ƿ�չʾ�м�ͼƬ
*/
Mat ImageP::LoGOperator(const string PicPath, bool show){
	Mat img = imread(PicPath);
	if (img.empty()){
		cout << "Cannot load image: " << PicPath << endl;
	}
	Mat imgGussian, img16S, imgLoG, imgSobelx, imgSobely, imgSobel,imgCanny;

	//LoG
	GaussianBlur(img, imgGussian, Size(3, 3), 1);
	Laplacian(imgGussian, img16S, 3);
	convertScaleAbs(img16S, imgLoG, 1);
	if (show){
		imshow("img before", img);
		imshow("LoG img", imgLoG);
		//waitKey(10);
	}
	//Sobel
	Sobel(img, img16S, 3, 1, 0);
	convertScaleAbs(img16S, imgSobelx, 1);
	Sobel(img, img16S, 3, 0, 1);
	convertScaleAbs(img16S, imgSobely, 1);
	add(imgSobelx, imgSobely, imgSobel);
	if (show){
		imshow("Sobel img", imgSobel);
		//waitKey(10);
	}
	//Canny
	Canny(img, imgCanny, 100, 200);
	if (show){
		imshow("Canny img", imgCanny);
		waitKey();
	}
	return imgLoG;
}
/*
@function: LBP������ȡͼƬ����
@PicPath: ͼƬ·��
@show: չʾͼƬ
*/
Mat ImageP::LBP(const string PicPath, bool show){
	Mat src = imread(PicPath,0);
	if (src.empty()){
		cout << "Cannot load img: " << PicPath << endl;
	}

	//Բ��LBP����
	Mat dst = Mat(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1, Scalar(0));;
	for (int n = 0; n<neighbors; n++)
	{
		// ������ļ���  
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// ��ȡ������ȡ����ֵ  
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// С������  
		float ty = y - fy;
		float tx = x - fx;
		// ���ò�ֵȨ��  
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// ѭ������ͼ������  
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				// �����ֵ  
				float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
				// ���б���  
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}

	//ԭʼ��LBP����
	Mat dst1 = Mat(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1, Scalar(0));
	// ѭ������ͼ������  
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar tt = 0;
			int tt1 = 0;
			uchar u = src.at<uchar>(i, j);
			if (src.at<uchar>(i - 1, j - 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j - 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
			tt1++;

			dst1.at<uchar>(i - 1, j - 1) = tt;// ������֮ǰ��dst.at<uchar>(i,j)=tt;  
		}
	}

	if (show){
		imshow("img before", src);
		imshow("circle", dst);
		imshow("normal", dst1);
		waitKey();
	}
	return dst;
}
/*
@function:����һάͼ���ֱ��ͼ
@param image:��Ҫ�任��ͼ��
@return:ֱ��ͼ
*/
Mat ImageP::Histogram1D(const Mat &image){
	histsize[0] = 256;
	hranges[0] = 0.0;
	hranges[1] = 255.0;
	ranges[0] = hranges;
	channels[0] = 0;

	MatND hist;
	calcHist(&image, 1, channels, Mat(), hist, 1, histsize, ranges);

	double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	Mat histImg(histsize[0], histsize[0], CV_8U, Scalar(255));
	int hpt = static_cast<int>(0.9*histsize[0]);
	for (int h = 0; h < histsize[0]; h++){
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, Point(h, histsize[0]), Point(h, histsize[0] - intensity), Scalar::all(0));
	}
	return histImg;
}
/*
@function: ��ȡͼƬ��ǰ��ͼ
@param PicPath:ͼƬ·��
@param show:�Ƿ�չʾ���
@return :ǰ��ͼ
*/
Mat ImageP::FrontSeg(const string PicPath, bool show){
	Mat src = imread(PicPath, 0);
	Mat BackGround = imread("C:\\Users\\Lenovo\\Desktop\\Mei_1.jpg", 0);
	Mat src1 = src - BackGround;
	if (src.empty()){
		cout << "Cannot load img: " << PicPath << endl;
	}
	Mat thresholded;
	threshold(src1, thresholded,5, 255, THRESH_BINARY_INV);

	if (show){
		imshow("image before", src);
		imshow("image no background", src1);
		imshow("Histogram", Histogram1D(src));
		imshow("thresholded", thresholded);
		waitKey();
	}

	return thresholded;
}
/*
@function:������˹�������ǽ����ڸ�˹�������Ļ����ϵģ����Ǹ�˹�������ڲ�ͬ��֮��Ĳ��
@param PicPath:ͼƬ·��
@param levels:�߶�
@param show:�Ƿ�չʾ���ͼ
@return :������˹�������²���ͼ��
*/
Mat ImageP::LaplacePyramid(const string PicPath, int levels, bool show){
	Mat src = imread(PicPath);
	Mat currentImg = src;
	Mat lap = currentImg;
	for (int l = 0; l < levels; l++){
		Mat up, down;
		pyrDown(currentImg, down);
		pyrUp(down, up, currentImg.size());
		lap = currentImg - up;
		currentImg = down;
	}
	if (show){
		imshow("image before", src);
		imshow("Laplace Pyramid", lap);
		waitKey();
	}
	return lap;
}
/*
@function:���ݵ������ص����ú̿�Ŀ��
@param PicPath:ͼƬ·��
@return�� ��Ȱٷֱ�
*/
double ImageP::CountWdith(const string refFramePath, const string curFramePath, bool show){
	Mat refFrame = imread(refFramePath, 0);// ����Ҷ�ͼ
	Mat curFrame = imread(curFramePath, 0);
	if (refFrame.empty() || curFrame.empty()){
		cout << "Cannot load Frame " << endl;
	}
	Mat procFrame;
	//ͼ�����
	procFrame = curFrame - refFrame;
	//�²���
	pyrDown(procFrame, procFrame, Size(procFrame.cols / 2, procFrame.rows / 2));

	if (show){
		imshow("Frame before", curFrame);
		imshow("pryDown", procFrame);
		waitKey();
	}
	return 0;








	
	//vector<float> countCols;
	//for (int i = img.rows/2; i < img.rows; i++){//��
	//	int count = 0;
	//	for (int j = 0; j < img.cols; j++){
	//		if (img.at<uchar>(i, j)> 0){//��
	//			count++;
	//		}
	//		//cout << i << "  " << count << endl;
	//		countCols.push_back((float)count * img.rows / (img.cols* i));
	//	}
	//}
	//double sum= 0;
	//for (int ic = 0; ic < countCols.size(); ic++){
	//	sum += countCols[ic];
	//}
	//sum = sum * 100 / countCols.size();
	//cout << "Mean Count is: " << sum <<"%"<< endl;
	//return sum;
}
/*
@function: ��ͼ��ӽ�������
@param PicPath: ͼƬ·��
@param n :��������Ŀ
@show :�Ƿ�չʾ���
@return :����ͼ
*/
Mat ImageP::AddSaltNoise(const string PicPath, int n, bool show){
	Mat srcImage = imread(PicPath);
	if (srcImage.empty()){
		cout << "Cannot load image: " << PicPath << endl;
	}
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;       //������  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;     //������  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	if (show){
		imshow("image before", srcImage);
		imshow("salt image", dstImage);
		waitKey();
	}
	return dstImage;
}
/*
@function:��˹����ֵ����ֵ��˫���˲�
@param Image: ��Ҫ�����ͼ��
@param show:�Ƿ�չʾͼ��
*/
void ImageP::Blur(const Mat &Image, bool show){
	Mat src = Image;
	//��˹�˲�
	//src:����ͼ��
	//dst:���ͼ��
	//Size(5,5)ģ���С��Ϊ����
	//x���򷽲�
	//Y���򷽲�
	Mat dstGauss = src.clone();
	GaussianBlur(src, dstGauss, Size(5, 5), 0, 0);

	//��ֵ�˲�
	//src:����ͼ��
	//dst::���ͼ��
	//ģ���ȣ�Ϊ����
	Mat dstMedian = src.clone();
	medianBlur(src, dstMedian, 3);

	//��ֵ�˲�
	//src:����ͼ��
	//dst:���ͼ��
	//ģ���С
	//Point(-1,-1):��ƽ����λ�ã�Ϊ��ֵȡ������
	Mat dstMean = src.clone();
	blur(src, dstMean, Size(3, 3), Point(-1, -1));

	//˫���˲�
	//src:����ͼ��
	//dst:����ͼ��
	//�˲�ģ��뾶
	//��ɫ�ռ��׼��
	//����ռ��׼��
	Mat dstBilater = src.clone();
	bilateralFilter(src, dstBilater, 5, 10.0, 2.0);//�����˲�ûʲôЧ����������

	if (show){
		imshow("image before", src);
		imshow("GaussianBlur",dstGauss);
		imshow("medianBlur", dstMedian);
		imshow("meanBlur", dstMean);
		imshow("bilateralFilter", dstBilater);
		waitKey();
	}

	waitKey();

}