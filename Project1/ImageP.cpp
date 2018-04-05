#include"ImageP.h"

/*
用于微信小程序大家来找茬
输入：
PicPath:输入文件路径
outpath:结果输出路径
show:是否展示中间运行结果图
输出：
结果图
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
	cout << "参考结果保存于：" << OutPath << endl;

	cv::resize(ROI_1, ROI_1, cv::Size(500, 500));
	cv::resize(ROI_2, ROI_2, cv::Size(500, 500));
	while (show){//以视频形式直观展出

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
利用sift特征点匹配两幅图片
输入：
PicPath_1:输入图片1位置
PicPath_2:输入图片2位置
OutPath:输出位置
show:是否展示中间过程
输出：
结果输出图
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
	//sift特征点检测
	SiftFeatureDetector siftdtc;
	vector<KeyPoint> kp1, kp2;
	siftdtc.detect(img1, kp1);
	Mat outimg1;
	drawKeypoints(img1, kp1, outimg1);
	siftdtc.detect(img2, kp2);
	Mat outimg2;
	drawKeypoints(img2, kp2, outimg2);
	//生成描述子
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
	cout << "结果保存于：" << OutPath << endl;
	
	if (show){
		imshow("desc", descp1);
		imshow("matches", img_matches);
		waitKey();
	}

	return img_matches;
	
}
/*
@function: surf特征点提取
@param PicPath:图片输入路径
@param show:是否展示中间结果
@return 该图像的surf特征图
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
@function: 利用图像中的hog特征，标出人物
@param PicPath:输入图片路径
@param show:是否展示中间结果
@return 标出的结果图
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
@function: 首先对图像进行高斯暖卷积滤波进行降噪处理，再采用Laplace算子进行边缘检测，就可以提高算子对噪声和离散点的Robust, 这一个过程中Laplacian of Gaussian(LOG)算子就诞生了
			函数中也给出了高斯变换，Sobel, Canny变换
@PicPath: 图片路径
@show:是否展示中间图片
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
@function: LBP算子提取图片特征
@PicPath: 图片路径
@show: 展示图片
*/
Mat ImageP::LBP(const string PicPath, bool show){
	Mat src = imread(PicPath,0);
	if (src.empty()){
		cout << "Cannot load img: " << PicPath << endl;
	}

	//圆形LBP算子
	Mat dst = Mat(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1, Scalar(0));;
	for (int n = 0; n<neighbors; n++)
	{
		// 采样点的计算  
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// 上取整和下取整的值  
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// 小数部分  
		float ty = y - fy;
		float tx = x - fx;
		// 设置插值权重  
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// 循环处理图像数据  
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				// 计算插值  
				float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
				// 进行编码  
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}

	//原始的LBP算子
	Mat dst1 = Mat(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1, Scalar(0));
	// 循环处理图像数据  
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

			dst1.at<uchar>(i - 1, j - 1) = tt;// 更正，之前是dst.at<uchar>(i,j)=tt;  
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
@function:画出一维图像的直方图
@param image:需要变换的图像
@return:直方图
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
@function: 提取图片的前景图
@param PicPath:图片路径
@param show:是否展示结果
@return :前景图
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
@function:拉普拉斯金字塔是建立在高斯金字塔的基础上的，就是高斯金字塔在不同层之间的差分
@param PicPath:图片路径
@param levels:尺度
@param show:是否展示结果图
@return :拉普拉斯金字塔下采样图像
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
@function:根据单个像素点计算煤炭的宽度
@param PicPath:图片路径
@return： 宽度百分比
*/
double ImageP::CountWdith(const string refFramePath, const string curFramePath, bool show){
	float threshod_1 = 15, threshod_2 = 45;
	int maskLen = 9;
	int beltWidth = 386;
	int widthBaseline = 400, widthThreshod = 255;

	Mat refFrame = imread(refFramePath, 0);// 读入灰度图
	Mat curFrame = imread(curFramePath, 0);
	if (refFrame.empty() || curFrame.empty()){
		cout << "Cannot load Frame " << endl;
	}
	Mat procFrame;
	//图像相减
	//procFrame = curFrame - refFrame;
	procFrame = refFrame - curFrame;
	//下采样
	pyrDown(procFrame, procFrame, Size(procFrame.cols / 2, procFrame.rows / 2));



	//二值化,结果为grayFrame
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

	//加入mask 卷积操作
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

	//计算宽度
	if (widthBaseline > nh)
		widthBaseline = (int)(nh / 3) * 2;
	int widthMiddle = (int)(nw / 2);
	//在图像中展示宽度基线和中线,用于调试
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
		cout << "宽度占比：" << result << endl;
		waitKey();
	}

	return result;


	//vector<float> countCols;
	//for (int i = img.rows/2; i < img.rows; i++){//行
	//	int count = 0;
	//	for (int j = 0; j < img.cols; j++){
	//		if (img.at<uchar>(i, j)> 0){//列
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
@function: 给图像加椒盐噪声
@param PicPath: 图片路径
@param n :噪声点数目
@show :是否展示结果
@return :噪声图
*/
Mat ImageP::AddSaltNoise(const string PicPath, int n, bool show){
	Mat srcImage = imread(PicPath);
	if (srcImage.empty()){
		cout << "Cannot load image: " << PicPath << endl;
	}
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;       //盐噪声  
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
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;     //椒噪声  
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
@function:高斯、中值、均值、双边滤波
@param Image: 需要处理的图像
@param show:是否展示图像
*/
void ImageP::Blur(const Mat &Image, bool show){
	Mat src = Image;
	//高斯滤波
	//src:输入图像
	//dst:输出图像
	//Size(5,5)模板大小，为奇数
	//x方向方差
	//Y方向方差
	Mat dstGauss = src.clone();
	GaussianBlur(src, dstGauss, Size(5, 5), 0, 0);

	//中值滤波
	//src:输入图像
	//dst::输出图像
	//模板宽度，为奇数
	Mat dstMedian = src.clone();
	medianBlur(src, dstMedian, 3);

	//均值滤波
	//src:输入图像
	//dst:输出图像
	//模板大小
	//Point(-1,-1):被平滑点位置，为负值取核中心
	Mat dstMean = src.clone();
	blur(src, dstMean, Size(3, 3), Point(-1, -1));

	//双边滤波
	//src:输入图像
	//dst:输入图像
	//滤波模板半径
	//颜色空间标准差
	//坐标空间标准差
	Mat dstBilater = src.clone();
	bilateralFilter(src, dstBilater, 5, 10.0, 2.0);//这里滤波没什么效果，不明白

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
@function:利用霍夫变换检测直线
@param PicPath:图片路径
@param show:是否展示结果
@return:结果图
*/
Mat ImageP:: LineFind(const string PicPath, bool show ){
	/*设定参数*/
	// 直线对应的点参数向量     
	std::vector<cv::Vec4i> lines;
	//步长     
	double deltaRho(1);
	double deltaTheta(PI / 180);
	// 判断是直线的最小投票数     
	int minVote(80);
	// 判断是直线的最小长度     
	double minLength(100);
	// 同一条直线上点之间的距离容忍度     
	double maxGap(20);
	//画线颜色
	Scalar color = Scalar(255,0, 0);

	/*图像处理*/
	Mat src = imread(PicPath);
	Mat result;
	cvtColor(src, result, CV_BGRA2BGR);
	Mat contour;
	Canny(result, contour, 125, 350);
	HoughLinesP(contour, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);

	/*画线段*/
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
@function：OCR图片预处理-去除直线
@param PicPath:图片位置
@param show:是否展示图片
@return 处理后的结果
*/
Mat ImageP::RemoveLine(const string PicPath, bool show){
	Mat src = imread(PicPath, 0);
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

	                                        传统按行逐个像素处理去除直线

	***************************************************************************************/
	//int bPoint, count = 0;
	//int threCount = 1;
	//for (int i = 0; i < nh; i++){
	//	//按行去除直线
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
	加入mask卷积去去除直线
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
	擦除面积小于【15个像素】的小块儿
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
	return dst;
}
/*
@function:调用印刷体OCR库
@param PicPath:图片路径
@param show:展示结果
@return 识别的结果（string）
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
@function:用opencv实现将证件照蓝底转为白底
@param PicPath:图片路径
@param show:展示结果
@return：最终结果
*/
Mat ImageP::BackgroundTransfer(const string PicPath, bool show){
	Mat image = imread(PicPath);
	int Diff;
	int num_row = image.rows;
	int num_col = image.cols;
	//初步
	for (int r = 0; r < num_row; r++)
	{
		cv::Vec3b *data = image.ptr<cv::Vec3b>(r);
		for (int c = 0; c < num_col; c++)
		{
			Diff = data[c][0] - (data[c][1] + data[c][2]) / 2; //蓝色检测
			if (Diff > 60 && data[c][0]>150)//蓝色分量比GR分量的平均值高60且蓝色分量大于150
			{
				data[c][0] = 255;
				data[c][1] = 255;
				data[c][2] = 255;
			}
		}
	}
	//优化
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
	//去毛边
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
	//		if (count >= 7 && count <= 18) //说明是头发边缘，开始处理
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