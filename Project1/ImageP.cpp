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
Mat ImageP::FindDiff(const string PicPath, bool show , const string OutPath){
	cv::Mat image;
	image = cv::imread(PicPath);

	cv::Mat ROI_1 = image(cv::Rect(210, 170, 829, 826));
	cv::Mat ROI_2 = image(cv::Rect(210, 1031, 829, 826));

	cv::Mat result;
	cv::Mat result_1 = ROI_1 - ROI_2;
	cv::Mat result_2 = ROI_2 - ROI_1;
	cv::threshold(result_1, result_1, 10, 255, cv::THRESH_BINARY_INV);
	cv::threshold(result_2, result_2, 10, 255, cv::THRESH_BINARY_INV);

	result = 0.4*result_1 + 0.2*ROI_1 + 0.4*result_2;
	cv::imwrite(OutPath, result);
	cout << "�ο���������ڣ�" << OutPath << endl;

	cv::resize(ROI_1, ROI_1, cv::Size(500, 500));
	cv::resize(ROI_2, ROI_2, cv::Size(500, 500));
	while (show){//����Ƶ��ʽֱ��չ��

		cv::imshow("ResultShow", ROI_1);
		cv::waitKey(20);
		cv::imshow("ResultShow", ROI_2);
		cv::waitKey(20);

		//cv::namedWindow("ROI1");
		//cv::imshow("ROI1", ROI_1);
		//cv::namedWindow("ROI2");
		//cv::imshow("ROI2", ROI_2);
		//cv::imshow("Result Image", result);
		//cv::waitKey();
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
	//һ����Ԥ������ɾ��
	resize(img1, img1, Size(312 * 2, 416 * 2));
	resize(img2, img2, Size(312 * 2, 416 * 2));
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
	//Mat outimage_sift = img1;
	//drawKeypoints(img1, kp1, outimage_sift, Scalar(120, 0, 120), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//if (show){
	//	imshow("SIFT1", outimg1);
	//	waitKey(10);
	//}

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
		//imshow("desc", descp1);
		imshow("matches", img_matches);
		waitKey();
	}

	return img_matches;
	
}
/*
@function: surf��sift��������ȡ
@param PicPath:ͼƬ����·��
@param show:�Ƿ�չʾ�м���
@return ��ͼ���surf����ͼ
*/
Mat ImageP::SurfFea(const string PicPath, bool show){
	Mat image = imread(PicPath);
	//Mat image = MoneyROI(PicPath, false);
	//һ����Ԥ������ɾ��
	//resize(image, image, Size(312*2, 416*2));


	if (image.empty()){
		cout << "Cannot load image:" << PicPath << endl;
	}
	if (show){
		imshow("image before", image);
		waitKey(10);
	}
	//surf����
	Mat outimage_surf;
	vector<KeyPoint> keypoints;
	SurfFeatureDetector surf(3000);
	surf.detect(image, keypoints);
	drawKeypoints(image, keypoints, outimage_surf, Scalar(120, 0, 120), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	if (show){
		imshow("SURF", outimage_surf);
		waitKey(10);
	}
	Mat outimage_sift;
	vector<KeyPoint> keypoints_sift;
	SiftFeatureDetector sift;
	//keypoints_sift.resize(1000);
	sift.detect(image, keypoints_sift);
	drawKeypoints(image, keypoints, outimage_sift, Scalar(120, 0, 120), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	if (show){
		
		imshow("SIFT", outimage_sift);
		waitKey();
	}
	return outimage_surf;
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
	float threshod_1 = 15, threshod_2 = 45;
	int maskLen = 9;
	int beltWidth = 386;
	int widthBaseline = 400, widthThreshod = 255;

	Mat refFrame = imread(refFramePath, 0);// ����Ҷ�ͼ
	Mat curFrame = imread(curFramePath, 0);
	if (refFrame.empty() || curFrame.empty()){
		cout << "Cannot load Frame " << endl;
	}
	Mat procFrame;
	//ͼ�����
	//procFrame = curFrame - refFrame;
	procFrame = refFrame - curFrame;
	//�²���
	pyrDown(procFrame, procFrame, Size(procFrame.cols / 2, procFrame.rows / 2));



	//��ֵ��,���ΪgrayFrame
	int nw = procFrame.cols;
	int nh = procFrame.rows;
	Mat grayFrame(nh, nw, CV_8U,Scalar::all(0));
	for (int i = 0; i < nh; i++){
		for (int j = 0; j < nw; j++){
			if (procFrame.at<uchar>(i, j)> threshod_1){
				procFrame.at<uchar>(i, j) = 255;
				grayFrame.at<uchar>(i, j) = 1;
			}
			else
				procFrame.at<uchar>(i, j) = 0;
		}
	}
	//threshold(procFrame, procFrame, 5, 255, THRESH_BINARY_INV);

	//����mask �������
	Mat mask(maskLen, maskLen, CV_8U, Scalar::all(1));
	filter2D(grayFrame, grayFrame, grayFrame.depth(), mask);
	for (int i = 0; i < nh; i++){
		for (int j = 0; j < nw; j++){
			if (grayFrame.at<uchar>(i, j)> threshod_2){
				procFrame.at<uchar>(i, j) = 255;
			}
			else
				procFrame.at<uchar>(i, j) = 0;
		}
	}

	//������
	if (widthBaseline > nh)
		widthBaseline = (int)(nh / 3) * 2;
	int widthMiddle = (int)(nw / 2);
	//��ͼ����չʾ��Ȼ��ߺ�����,���ڵ���
	//for (int i = 0; i < nw; i++){
	//	procFrame.at<uchar>(widthBaseline, i) = 255;
	//	//procFrame.at<uchar>(i, widthBaseline) = 255;
	//}
	//for (int i = 0; i < nh; i++){
	//	procFrame.at<uchar>(i, widthMiddle) = 255;
	//}

	double result =0;
	for (int i = widthMiddle - 150; i < widthMiddle + 150; i++){
		if (procFrame.at<uchar>(widthBaseline, i) == widthThreshod)
			result++;
	}
	result = (result * 100) / beltWidth;

	if (show){
		imshow("Frame before", curFrame);
		imshow("Precossed", procFrame);
		imshow("Histogram1D", Histogram1D(procFrame));
		cout << "���ռ�ȣ�" << result << endl;
		waitKey();
	}

	return result;


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
/*
@function:���û���任���ֱ��
@param PicPath:ͼƬ·��
@param show:�Ƿ�չʾ���
@return:���ͼ
*/
Mat ImageP:: LineFind(const string PicPath, bool show ){
	/*�趨����*/
	// ֱ�߶�Ӧ�ĵ��������     
	std::vector<cv::Vec4i> lines;
	//����     
	double deltaRho(1);
	double deltaTheta(PI / 180);
	// �ж���ֱ�ߵ���СͶƱ��     
	int minVote(80);
	// �ж���ֱ�ߵ���С����     
	double minLength(200);
	// ͬһ��ֱ���ϵ�֮��ľ������̶�     
	double maxGap(10);
	//������ɫ
	Scalar color = Scalar(255,0, 0);

	/*ͼ����*/
	Mat src = imread(PicPath);
	Mat result;
	//�任��ɫ�ռ�
	//cvtColor(src, result, CV_BGRA2BGR);
	cvtColor(src, result, CV_BGR2GRAY);
	if (show){
		imshow("gray", result);
	}
	////��ֵ��
	//threshold(result, result, 20, 255, THRESH_BINARY);
	//if (show){
	//	imshow("threshold", result);
	//}
	Mat contour;
	Canny(result, contour, 125, 350);
	if (show){
		imshow("Canny", contour);
	}
	//�����㣨�����ͺ�ʴ�� ��С��
	//Mat element7(7, 7, CV_8U, cv::Scalar(1));
	Mat element7(5, 5, CV_8U, cv::Scalar(1));
	morphologyEx(contour, contour, MORPH_CLOSE, element7);
	if (show){
		imshow("morphology", contour);
	}

	HoughLinesP(contour, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);

	/*���߶�*/
	vector<Vec4i>::const_iterator it = lines.begin();
	while (it != lines.end()){
		Point pt1((*it)[0], (*it)[1]);
		Point pt2((*it)[2], (*it)[3]);
		line(src, pt1, pt2, color);
		++it;
	}
	if (show){
		imshow("lines", src);
		waitKey();
	}
	return src;
}
/*
@function��OCRͼƬԤ����-ȥ��ֱ��
@param PicPath:ͼƬλ��
@param show:�Ƿ�չʾͼƬ
@return �����Ľ��
*/
Mat ImageP::RemoveLine(Mat src, bool show){
	//Mat src = imread(PicPath, 0);
	if (src.empty()){
		cout << "Cannot load image src" << endl;
	}
	Mat dst;
	//pyrUp(src, src, Size(src.cols * 2, src.rows * 2));
	//blur(src, src, Size(3, 3), Point(-1, -1));
	src.copyTo(dst);

	int nw = src.cols;
	int nh = src.rows;

	/* **********************************************************************************

	                                        ��ͳ����������ش���ȥ��ֱ��

	***************************************************************************************/
	//int bPoint, count = 0;
	//int threCount = 1;
	//for (int i = 0; i < nh; i++){
	//	//����ȥ��ֱ��
	//	for (int j = 0; j < nw; j++){
	//		if (count == 0 && src.at<uchar>(i, j) == 0){
	//			bPoint = j;
	//			count++;
	//		}
	//		else if (src.at<uchar>(i, j) == 0){
	//			count++;
	//		}
	//		if (count > threCount &&  src.at<uchar>(i, j) != 0){
	//			for (int k = 0; k < count; k++)
	//				dst.at<uchar>(i, bPoint + k) = 255;
	//			count = 0;
	//			bPoint = 0;
	//		}
	//	}
	//}

	/***********************************************************************************
	����mask���ȥȥ��ֱ��
	*************************************************************************************/
	//int maskLen = 5;
	//Mat mask(maskLen, maskLen, CV_8U, Scalar::all(1));
	//Mat kern = (Mat_<char>(3, 3) << 1, 0, 1,
	//								1, 0, 1,
	//								1, 0, 1);
	//filter2D(dst, dst, dst.depth(), kern);
	//for (int i = 0; i < nh; i++){
	//	for (int j = 0; j < nw; j++){
	//		if (dst.at<uchar>(i, j)> 2){
	//			dst.at<uchar>(i, j) = 255;
	//		}
	//		else
	//			dst.at<uchar>(i, j) =0;
	//	}
	//}

	//pyrUp(dst, dst, Size(dst.cols * 2, dst.rows * 2));
	//if (show){
	//	imshow("src", src);
	//	imshow("dst", dst);
	//	waitKey();
	//}

	/**********************************************************************************
	�������С�ڡ�15�����ء���С���
	**********************************************************************************/
	pyrUp(dst, dst, Size(dst.cols * 2, dst.rows * 2));

	threshold(dst, dst, 120, 255, CV_THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	findContours(dst, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	dst = Scalar::all(0);
	drawContours(dst, contours, -1, Scalar::all(255));
	vector<vector<Point>>::iterator it = contours.begin();
	for (int i = 0; i < contours.size(); i++){
	//for (; it != contours.end; it++){
		//if (fabs(contourArea(*it)) < 15)
		if (fabs(contourArea(contours[i])) < 13)
			drawContours(dst, contours, i, Scalar(0), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
	}
	if (show){
			imshow("src", src);
			imshow("dst", dst);
			waitKey();
	}
	resize(dst, dst, Size(dst.cols / 2, dst.rows / 2));
	return dst;
}
/*
@function:����ӡˢ��OCR��
@param PicPath:ͼƬ·��
@param show:չʾ���
@return ʶ��Ľ����string��
*/
string ImageP::VilabOCR(const string PicPath, bool show){
	OCR_StatusError e;
	OCR_Handle handle = OCR_InitLib(&e);
	printf("OCR_InitLib:%s\n", OCR_GetError(&e));

	const char* filename = PicPath.c_str();
	string  output = ViLab_OCR(filename, &e);
	if (show){
		cout << output << endl << "finished!" << endl;
	}
	return output;
}
/*
@function:��opencvʵ�ֽ�֤��������תΪ�׵�
@param PicPath:ͼƬ·��
@param show:չʾ���
@return�����ս��
*/
Mat ImageP::BackgroundTransfer(const string PicPath, bool show){
	Mat image = imread(PicPath);
	int Diff;
	int num_row = image.rows;
	int num_col = image.cols;
	//����
	for (int r = 0; r < num_row; r++)
	{
		cv::Vec3b *data = image.ptr<cv::Vec3b>(r);
		for (int c = 0; c < num_col; c++)
		{
			Diff = data[c][0] - (data[c][1] + data[c][2]) / 2; //��ɫ���
			if (Diff > 60 && data[c][0]>150)//��ɫ������GR������ƽ��ֵ��60����ɫ��������150
			{
				data[c][0] = 255;
				data[c][1] = 255;
				data[c][2] = 255;
			}
		}
	}
	//�Ż�
	for (int i = 1; i < num_row - 1; i++)
	{
		cv::Vec3b *last_r = image.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b *data = image.ptr<cv::Vec3b>(i);
		cv::Vec3b *next_r = image.ptr<cv::Vec3b>(i + 1);
		for (int j = 1; j < num_col - 1; j++)
		{
			if (data[j][0]>90 && data[j][0] - data[j][1]>9 && data[j][0] - data[j][2]>9)
			{
				int stat;
				cv::Vec3b Temp;
				cv::Vec3b array[9] = { last_r[j - 1], last_r[j], last_r[j + 1], data[j - 1], data[j], data[j + 1], next_r[j - 1], next_r[j], next_r[j + 1] };
				do
				{
					stat = 0;
					for (int m = 0; m < 8; m++)
					{
						if (array[m][0] + array[m][1] + array[m][2]> array[m + 1][0] + array[m + 1][1] + array[m + 1][2])
						{
							Temp = array[m + 1];
							array[m + 1] = array[m];
							array[m] = Temp;
							stat = 1;
						}

					}
				} while (stat == 1);
				data[j][0] = array[7][0];
				data[j][1] = array[7][1];
				data[j][2] = array[7][2];
			}
		}
	}
	//ȥë��
	//for (int i = 2; i < num_row - 2; i = i + 5)
	//{
	//	cv::Vec3b *last_sec_r = image.ptr<cv::Vec3b>(i - 2);
	//	cv::Vec3b *last_r = image.ptr<cv::Vec3b>(i - 1);
	//	cv::Vec3b *data = image.ptr<cv::Vec3b>(i);
	//	cv::Vec3b *next_r = image.ptr<cv::Vec3b>(i + 1);
	//	cv::Vec3b *next_sec_r = image.ptr<cv::Vec3b>(i + 2);
	//	for (int j = 2; j < num_col; j = j + 5)
	//	{
	//		int count = 0;// check how many 255point in this area(boundary)
	//		cv::Vec3b array[5][5] = { last_sec_r[j - 2], last_sec_r[j - 1], last_sec_r[j], last_sec_r[j + 1], last_sec_r[j + 2], \
	//			last_r[j - 2], last_r[j - 1], last_r[j], last_r[j + 1], last_r[j + 2], \
	//			data[j - 2], data[j - 1], data[j], data[j + 1], data[j + 2], \
	//			next_r[j - 2], next_r[j - 1], next_r[j], next_r[j + 1], next_r[j + 2], \
	//			next_sec_r[j - 2], next_sec_r[j - 1], next_sec_r[j], next_sec_r[j + 1], next_sec_r[j + 2] };
	//		for (int r = 0; r < 5; r++)
	//		{
	//			for (int c = 0; c < 5; c++)
	//			{
	//				if (array[r][c][1] >= 251)
	//					count++;
	//			}
	//		}
	//		if (count >= 7 && count <= 18) //˵����ͷ����Ե����ʼ����
	//		{

	//			last_r[j - 1] = 1 / 9 * (array[0][0] + array[0][1] + array[0][2] + array[1][0] + array[1][1] + array[1][2] + array[2][0] + array[2][1] + array[2][2]) + cv::Vec3b(100, 100, 100);
	//			last_r[j] = 1 / 9 * (array[0][1] + array[0][2] + array[0][3] + array[1][1] + array[1][2] + array[1][3] + array[2][1] + array[2][2] + array[2][3]) + cv::Vec3b(80, 80, 80);
	//			last_r[j + 1] = 1 / 9 * (array[0][2] + array[0][3] + array[0][4] + array[1][2] + array[1][3] + array[1][4] + array[2][2] + array[2][3] + array[2][4]) + cv::Vec3b(100, 100, 100);

	//			data[j - 1] = (1 / 9 * array[1][0] + 1 / 9 * array[1][1] + 1 / 9 * array[1][2] + 1 / 9 * array[2][0] + 1 / 9 * array[2][1] + 1 / 9 * array[2][2] + 1 / 9 * array[3][0] + 1 / 9 * array[3][1] + 1 / 9 * array[3][2]) + cv::Vec3b(80, 80, 80);
	//			data[j] = (1 / 9 * array[1][1] + 1 / 9 * array[1][2] + 1 / 9 * array[1][3] + 1 / 9 * array[2][1] + 1 / 9 * array[2][2] + 1 / 9 * array[2][3] + 1 / 9 * array[3][1] + 1 / 9 * array[3][2] + 1 / 9 * array[3][3]) + cv::Vec3b(80, 80, 80);
	//			data[j + 1] = (1 / 9 * array[1][2] + 1 / 9 * array[1][3] + 1 / 9 * array[1][4] + 1 / 9 * array[2][2] + 1 / 9 * array[2][3] + 1 / 9 * array[2][4] + 1 / 9 * array[3][2] + 1 / 9 * array[3][3] + 1 / 9 * array[3][4]) + cv::Vec3b(80, 80, 80);

	//			data[j - 1] = (1 / 9 * array[2][0] + 1 / 9 * array[2][1] + 1 / 9 * array[2][2] + 1 / 9 * array[3][0] + 1 / 9 * array[3][1] + 1 / 9 * array[3][2] + 1 / 9 * array[4][0] + 1 / 9 * array[4][1] + 1 / 9 * array[4][2]) + cv::Vec3b(100, 100, 100);
	//			data[j] = (1 / 9 * array[2][1] + 1 / 9 * array[2][2] + 1 / 9 * array[2][3] + 1 / 9 * array[3][1] + 1 / 9 * array[3][2] + 1 / 9 * array[3][3] + 1 / 9 * array[4][1] + 1 / 9 * array[4][2] + 1 / 9 * array[4][3]) + cv::Vec3b(80, 80, 80);
	//			data[j + 1] = (1 / 9 * array[2][2] + 1 / 9 * array[2][3] + 1 / 9 * array[2][4] + 1 / 9 * array[3][2] + 1 / 9 * array[3][3] + 1 / 9 * array[3][4] + 1 / 9 * array[4][2] + 1 / 9 * array[4][3] + 1 / 9 * array[4][4]) + cv::Vec3b(100, 100, 100);

	//		}
	//	}
	//}
	if (show){
		imshow("Pic", image);
		waitKey();
	}
	return image;
}
/*
���˵Ĵ��룬�����������
*/
void ImageP::GetContoursPic(const string pSrcFileName, const string pDstFileName){
	Mat srcImg = imread(pSrcFileName);
	Mat gray, binImg;
	//�ҶȻ�
	cvtColor(srcImg, gray, COLOR_RGB2GRAY);
	imshow("�Ҷ�ͼ", gray);
	//��ֵ��
	threshold(gray, binImg, 100, 200, CV_THRESH_BINARY);
	imshow("��ֵ��", binImg);

	vector<vector<Point> > contours;
	vector<Rect> boundRect(contours.size());
	//ע���5������ΪCV_RETR_EXTERNAL��ֻ�������  
	findContours(binImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //������
	cout << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		//��Ҫ��ȡ������  
		CvPoint2D32f rectpoint[4];
		CvBox2D rect = minAreaRect(Mat(contours[i]));

		cvBoxPoints(rect, rectpoint); //��ȡ4����������  
		//��ˮƽ�ߵĽǶ�  
		float angle = rect.angle;
		cout << angle << endl;

		int line1 = sqrt((rectpoint[1].y - rectpoint[0].y)*(rectpoint[1].y - rectpoint[0].y) + (rectpoint[1].x - rectpoint[0].x)*(rectpoint[1].x - rectpoint[0].x));
		int line2 = sqrt((rectpoint[3].y - rectpoint[0].y)*(rectpoint[3].y - rectpoint[0].y) + (rectpoint[3].x - rectpoint[0].x)*(rectpoint[3].x - rectpoint[0].x));
		//rectangle(binImg, rectpoint[0], rectpoint[3], Scalar(255), 2);
		//���̫С��ֱ��pass
		if (line1 * line2 < 600)
		{
			continue;
		}

		//Ϊ���������κ��ŷţ�������ת�Ƕ��ǲ�һ���ġ����ŵģ�������90�ȣ�������  
		if (line1 > line2)
		{
			angle = 90 + angle;
		}

		//�½�һ������Ȥ������ͼ����С��ԭͼһ����  
		Mat RoiSrcImg(srcImg.rows, srcImg.cols, CV_8UC3); //ע���������ѡCV_8UC3
		RoiSrcImg.setTo(0); //��ɫ������Ϊ��ɫ  
		//imshow("�½���ROI", RoiSrcImg);
		//�Եõ����������һ��  
		drawContours(binImg, contours, -1, Scalar(255), CV_FILLED);

		//��ͼ��RoiSrcImg
		srcImg.copyTo(RoiSrcImg, binImg);


		//����ʾһ�¿��������˸���Ȥ�������������ֶ��Ǻ�ɫ����  
		namedWindow("RoiSrcImg", 1);
		imshow("RoiSrcImg", RoiSrcImg);

		//����һ����ת���ͼ��  
		Mat RatationedImg(RoiSrcImg.rows, RoiSrcImg.cols, CV_8UC1);
		RatationedImg.setTo(0);
		//��RoiSrcImg������ת  
		Point2f center = rect.center;  //���ĵ�  
		Mat M2 = getRotationMatrix2D(center, angle, 1);//������ת�����ŵı任���� 
		warpAffine(RoiSrcImg, RatationedImg, M2, RoiSrcImg.size(), 1, 0, Scalar(0));//����任 
		imshow("��ת֮��", RatationedImg);
		imwrite("r.jpg", RatationedImg); //���������ͼƬ��������
	}

#if 1
	//��ROI������п�ͼ

	//����ת���ͼƬ����������ȡ  
	vector<vector<Point> > contours2;
	Mat raw = imread("r.jpg");
	Mat SecondFindImg;
	//SecondFindImg.setTo(0);
	cvtColor(raw, SecondFindImg, COLOR_BGR2GRAY);  //�ҶȻ�  
	threshold(SecondFindImg, SecondFindImg, 80, 200, CV_THRESH_BINARY);
	findContours(SecondFindImg, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//cout << "sec contour:" << contours2.size() << endl;

	for (int j = 0; j < contours2.size(); j++)
	{
		//��ʱ����ʵ����һ���������ˣ����Ի�ȡrect  
		Rect rect = boundingRect(Mat(contours2[j]));
		//���̫С������ֱ��pass,ͨ�����ù��������С�����Ա�ֻ֤�õ����
		if (rect.area() < 600)
		{
			continue;
		}
		Mat dstImg = raw(rect);
		imshow("dst", dstImg);
		imwrite(pDstFileName, dstImg);
	}
#endif
}
/*
function:�����찵�����µ�ֽ�ң�����ֽ������ ���Ľ�������10Ԫֽ�ң�����Ч�������C5Ч������
return :���������ͼƬ
*/
Mat ImageP::MoneyROI(const string PicPath, bool show){
	Mat img = imread(PicPath);


	//RGBת��ΪHSV�ñ��Ͷ���ȡֽ�ҵ�����
	Mat imgHSV;
	vector<Mat>v;
	resize(img, img, Size(img.cols / 4, img.rows / 4));
	if (show)
		imshow("src", img);
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	split(imgHSV, v);
	//ȡ���Ͷ�
	threshold(v[1], v[1], 20, 255, THRESH_BINARY);
	
	//ȥ��ë�ߣ����п����㣬�ȸ�ʴ������
	Mat element7(7, 7, CV_8U, cv::Scalar(1));
	morphologyEx(v[1], v[1], MORPH_OPEN, element7);

	if (show)
		imshow("���Ͷ�", v[1]);
	//Processor.Blur(img);

	//��ȡ����
	vector<vector<Point>>contours;
	findContours(v[1], contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector<Rect> roiAll;
	Mat result(img);

	//drawContours(img, contours, -1, Scalar(0), 2);
	//�Ƴ������С������
	int cmin = 100;
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()){
		if (itc->size() < cmin)
			itc = contours.erase(itc);
		else{
			//�����ο�
			Rect r = boundingRect(Mat(*itc));
			roiAll.push_back(r);
			rectangle(result, r, Scalar(255), 2);
			++itc;
		}

	}
	
	//merge all Rect
	//groupRectangles(roiAll, 5, 7);
	Rect res;
	//�����ں�
	res = GroupRect(roiAll);
	rectangle(result, res, Scalar(0, 0, 255), 2);


	if (show){
		imshow("front", result);
		waitKey(0);
	}
	
	//����ROI

	return result(res);
}
/*
ROI�����ں�
*/
Rect ImageP::GroupRect(vector<Rect>RectList){
	Rect res;
	vector<Rect>::iterator it = RectList.begin();
	res = *it++;
	while (it != RectList.end()){
		float x = res.x, y = res.y;
		res.x = it->x < x ? it->x : x;
		res.y = it->y < y ? it->y : y;
		//����Ϊwidth,����Ϊheight
		float height_1 = it->y + it->height;
		float height_2 = y + res.height;
		res.height = height_1 > height_2 ? height_1 - it->y : height_2 - it->y;
		float weight_1 = it->x + it->width;
		float weight_2 = x + res.width;
		res.width = weight_1 > weight_2 ? weight_1 - it->x : weight_2 - it->x;
		it++;
	}
	return res;
}
/*
ģ��ƥ�䵥������
*/
void ImageP::SingleTemplateMatch(const string PicPath, const string TemplPath, bool show){
	//IplImage*src, *templat, *result;
	int srcW, templatW, srcH, templatH;
	//����Դͼ��  
	Mat src = imread(PicPath, 0);

	//����ģ��ͼ��  
	Mat templat = imread(TemplPath, 0);

	if (!src.data || !templat.data){
		printf("��ͼƬʧ��");
	}


	//����������Ĵ�С  
	int result_cols = src.cols - templat.cols + 1;
	int result_rows = src.rows - templat.rows + 1;

	//������Ž���Ŀռ�  
	Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	double minVal, maxVal;
	Point minLoc; Point maxLoc;

	//����ģ��ƥ�亯�� 
	matchTemplate(src, templat, result, CV_TM_SQDIFF);

	//���������Ƶ�ֵ������������  
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());


	printf("minVal  %f   maxVal %f\n ", minVal, maxVal);

	//���ƽ��   
	rectangle(src, minLoc, cvPoint(minLoc.x + templat.cols, minLoc.y + templat.rows), Scalar::all(0), 2, 8, 0);

	//��ʾ���  
	if (show){
		imshow("show", src);
		imshow("tem", templat);
		cvWaitKey(0);
	}

}
/*
ģ��ƥ��������
*/
void ImageP::MultiTemplateMatch(const string  PicPath, const string TemplPath, bool show){
	Mat srcImg = imread(PicPath);
	Mat templateImg = imread(TemplPath);
	Mat resultImg;
	Mat showImg = srcImg.clone();

	int resultImg_cols = srcImg.cols - templateImg.cols + 1;
	int resultImg_rows = srcImg.rows - templateImg.rows + 1;

	resultImg.create(resultImg_cols, resultImg_rows, CV_32FC1);
	matchTemplate(srcImg, templateImg, resultImg, CV_TM_CCOEFF_NORMED); //�����ϵ��ƥ�䷨(���ƥ��1)
	normalize(resultImg, resultImg, 0, 1, NORM_MINMAX);
	Mat midImg = resultImg.clone();

	//��Ŀ��ģ��ƥ��---����һ
	double matchValue;
	int count0=0;
	int tempW=0, tempH=0;
	char matchRate[10];

	for(int i=0; i<resultImg_rows; i++)
	{
		for(int j=0; j<resultImg_cols; j++)
		{
			matchValue = resultImg.at<float>(i, j);
			sprintf(matchRate, "%0.2f", matchValue);
			if(matchValue>=0.85 && (abs(j - tempW)>5) && (abs(i - tempH)>5) )
			{
				count0++;
				putText(showImg, matchRate, Point(j-5, i-5), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1);
				rectangle(showImg, Point(j, i), Point(j + templateImg.cols, i + templateImg.rows), Scalar(0, 255, 0), 2);
				tempW = j;
				tempH = i;
			}
		}
	}
	cout<<"count="<<count0<<endl;
	if (show){
		imshow("resultImg", resultImg);
		imshow("dst", showImg);
		waitKey();
	}

	//�ڶ��ַ���
	//double minValue, maxValue;
	//Point minLoc, maxLoc;
	//Point matchLoc;
	//char matchRate[10];

	//for (int i = 0; i<100; i++)
	//{
	//	int startX = maxLoc.x - 4;
	//	int startY = maxLoc.y - 4;
	//	int endX = maxLoc.x + 4;
	//	int endY = maxLoc.y + 4;
	//	if (startX<0 || startY<0)
	//	{
	//		startX = 0;
	//		startY = 0;
	//	}
	//	if (endX > resultImg.cols - 1 || endY > resultImg.rows - 1)
	//	{
	//		endX = resultImg.cols - 1;
	//		endY = resultImg.rows - 1;
	//	}
	//	Mat temp = Mat::zeros(endX - startX, endY - startY, CV_32FC1);
	//	//Mat ROI = resultImg(Rect(Point(startX, startY), temp.cols, temp.rows));
	//	temp.copyTo(resultImg(Rect(startX, startY, temp.cols, temp.rows)));
	//	minMaxLoc(resultImg, &minValue, &maxValue, &minLoc, &maxLoc);
	//	if (maxValue<0.8)    break;

	//	cout << "max_value= " << maxValue << endl;
	//	sprintf(matchRate, "%0.2f", maxValue);
	//	putText(showImg, matchRate, Point(maxLoc.x - 5, maxLoc.y - 5), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1);
	//	rectangle(showImg, maxLoc, Point(maxLoc.x + templateImg.cols, maxLoc.y + templateImg.rows), Scalar(0, 255, 0), 2);

	//}
	//if (show){
	//	imshow("midImg", midImg);
	//	imshow("resultImg", resultImg);
	//	imshow("dst", showImg);
	//	waitKey(0);
	//}


}
/*
@function:����ݶ�ͼ
@param PicPath : ͼƬ·��
@param show : չʾ���
@return�����ս��
*/
Mat	ImageP::Gradient(const string PicPath, bool show){
	Mat picture = imread(PicPath, 0);//�Ҷ�  
	
	//Mat picture = MoneyROI(PicPath, false);
	//cvtColor(picture, picture, CV_BGR2GRAY);

	Mat img;

	picture.convertTo(img, CV_32F); //ת���ɸ���  
	sqrt(img, img);                 //gammaУ��  
	normalize(img, img, 0, 255, NORM_MINMAX, CV_32F);//��һ��[0,255]������  

	Mat gradient = Mat::zeros(img.rows, img.cols, CV_32F);//�ݶ�  
	Mat theta = Mat::zeros(img.rows, img.cols, CV_32F);//�Ƕ�  

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			float Gx, Gy;

			Gx = img.at<float>(i, j + 1) - img.at<float>(i, j - 1);
			Gy = img.at<float>(i + 1, j) - img.at<float>(i - 1, j);

			gradient.at<float>(i, j) = sqrt(Gx * Gx + Gy * Gy);//�ݶ�ģֵ  
			theta.at<float>(i, j) = float(atan2(Gy, Gx) * 180 / CV_PI);//�ݶȷ���[-180�㣬180��]  
		}
	}

	normalize(gradient, gradient, 0, 255, NORM_MINMAX, CV_8UC1);//��һ��[0,255] �޷�������  
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
	if (show){
		imshow("ԭͼ", picture);
		//imshow("GammaУ��", img);
		imshow("�ݶ�ͼ", gradient);
		waitKey();
	}

	return gradient;
}
/*
��ʾ����λ��--����¼� ������ȫ�ֺ���
*/
void PiexLocation_Mouse(int EVENT, int x, int y, int flags, void* userdata){

	Mat hh;
	hh = *(Mat*)userdata;
	char Loc[16];
	Point p(x, y);
	switch (EVENT){
		case EVENT_LBUTTONDOWN:{
			printf("(%d, %d)", x, y);
			printf("b=%d\t", hh.at<Vec3b>(p)[0]);
			printf("g=%d\t", hh.at<Vec3b>(p)[1]);
			printf("r=%d\n", hh.at<Vec3b>(p)[2]);
			circle(hh, p, 2, Scalar(255), 3);

			sprintf(Loc, "(%d,%d)", x, y);
			putText(hh, Loc, p, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255),1);
		}
		break;
	}
}
/*
��ʾ����λ��
*/
void ImageP::PiexLocation_Show(const string PicPath){
	namedWindow("display");
	Mat src = imread(PicPath);
	setMouseCallback("display", PiexLocation_Mouse, &src);
	while (1){
		imshow("display", src);
		waitKey(40);
	}
}
/*
��������
�÷�:
ImageP Processor;
string PicPath1 = "C:\\Users\\Lenovo\\Desktop\\mei.png";
Mat img = imread(PicPath1, 0);
vector<Mat>wnd;
Processor.SlidingWnd(img, wnd);
imshow("src", img);
waitKey(0);
*/
Mat ImageP::SlidingWnd(Mat &src, vector<Mat> &wnd, int n/*, Size &wndSize, double x_percent, double y_percent*/){
	int count = 0;//��¼�������ڵ���Ŀ 
	//int x_tep = cvCeil(x_percent *wndSize.width);
	//int y_tep = cvCeil(y_percent *wndSize.height);
	Mat origin = src.clone();
	cvtColor(src, src, CV_BGR2GRAY);

	//ԭͼ�ϻ���
	for (int i = 0; i < src.cols - src.cols / n + 1; i += src.cols / n){//����
		Point start = Point(i, 0);
		Point end = Point(i, src.rows);
		line(origin, start, end, Scalar(0, 255, 0));
	}
	for (int j = 0; j < src.rows - src.rows / n + 1; j += src.rows / n){//����
		Point start = Point(0, j);
		Point end = Point(src.cols, j);
		line(origin, start, end, Scalar(0, 255, 0));
	}
	Mat Mean, Stddv;//��ֵ�ͷ���

	char wndName[] = "C:\\Users\\Lenovo\\Desktop\\temp\\tmp\\";
	char temp[1000];
	for (int i = 0; i < src.cols - src.cols / n + 1; i += src.cols / n){
		for (int j = 0; j < src.rows - src.rows / n + 1; j += src.rows / n){
			count++;
			//Rect roi(Point(j, i), wndSize);
			cout << "�� " << count << "������" << " ";
			//cout << "x=" << i <<" ";
			//cout << "y=" << j <<" ";
			Mat ROI = src(Rect(i, j, src.cols / n, src.rows / n));
			wnd.push_back(ROI);


			//���������ֵ�ͷ���
			meanStdDev(ROI, Mean, Stddv);
			char meanStr[10];
			char stddvStr[10];
			string str = to_string(Mean.at<double>(0, 0)) + "  " + to_string(Stddv.at<double>(0, 0)) + " " +to_string(CalMeanGrad(ROI));
			cout <<"Mean Std  Grad = "<< str << endl;
			
			putText(origin, to_string(count), Point(i + src.cols / (2 * n), j + src.rows / (2 * n)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
			//putText(src, stddvStr, Point(i, j), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

			sprintf_s(temp, "%s%d%s", wndName, count, ".jpg");
			imwrite(temp, ROI);
		}
	}
	cout << count << endl;
	return origin;
}
double ImageP::CalMeanGrad(Mat img){
	//cvtColor(src, img, CV_RGB2GRAY); // ת��Ϊ�Ҷ�ͼ
	img.convertTo(img, CV_64FC1);
	double tmp = 0;
	int rows = img.rows - 1;
	int cols = img.cols - 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double dx = img.at<double>(i, j + 1) - img.at<double>(i, j);
			double dy = img.at<double>(i + 1, j) - img.at<double>(i, j);
			double ds = std::sqrt((dx*dx + dy*dy) / 2);
			tmp += ds;
		}
	}
	double imageAvG = tmp / (rows*cols);
	return imageAvG;
}