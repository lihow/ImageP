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
Mat ImageP::FindDiff(const string PicPath, const string OutPath, bool show){
	cv::Mat image;
	image = cv::imread(PicPath);

	cv::Mat ROI_1 = image(cv::Rect(210, 170, 829, 826));
	cv::Mat ROI_2 = image(cv::Rect(210, 1031, 829, 826));

	cv::Mat result = ROI_1 - ROI_2;
	cv::threshold(result, result, 10, 255, cv::THRESH_BINARY_INV);
	result = 0.8*result + 0.2*ROI_1;
	cv::imwrite(OutPath, result);
	cout << "结果保存于：" << OutPath << endl;

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
	Mat refFrame = imread(refFramePath, 0);// 读入灰度图
	Mat curFrame = imread(curFramePath, 0);
	if (refFrame.empty() || curFrame.empty()){
		cout << "Cannot load Frame " << endl;
	}
	Mat procFrame;
	//图像相减
	procFrame = curFrame - refFrame;
	//下采样
	pyrDown(procFrame, procFrame, Size(procFrame.cols / 2, procFrame.rows / 2));

	if (show){
		imshow("Frame before", curFrame);
		imshow("pryDown", procFrame);
		waitKey();
	}
	return 0;








	
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