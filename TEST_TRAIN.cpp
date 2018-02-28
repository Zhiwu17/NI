#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\ml\ml.hpp>

#include<iostream>
#include<sstream>

using namespace std;
using namespace cv;
using namespace cv::ml;


const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

const int MIN_PIXEL_WIDTH = 20;
const int MIN_PIXEL_HEIGHT = 20;

const double MIN_ASPECT_RATIO = 0.25;
const double MAX_ASPECT_RATIO = 1.0;

class ContourWithData
{
public:
	ContourWithData();
	~ContourWithData();
	std::vector<cv::Point> ptContour;
	cv::Rect boundingRect;
	float fltArea;

	bool checkIfCotourIsValid(){
		if (fltArea > MIN_CONTOUR_AREA && boundingRect.width > MIN_PIXEL_WIDTH && boundingRect.height > MIN_PIXEL_HEIGHT) 
			return true;
		return false;
	}
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight){
		return (cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}

private:

};

ContourWithData::ContourWithData()
{
}

ContourWithData::~ContourWithData()
{
}

int main()
{
	std::vector<ContourWithData> allContoursWithData;
	std::vector<ContourWithData> validContoursWithData;

	cv::Mat matClassificationFloats;

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);

	if (fsClassifications.isOpened() == false){
		std::cout << "error, unable to open training classifications file, extiting program\n\n";
		return(0);
	}

	fsClassifications["classifications"] >> matClassificationFloats;
	fsClassifications.release();

	cv::Mat matTrainingImages;

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);

	if (fsTrainingImages.isOpened() == false){
		std::cout << "error, unable to open training images files, exiting program\n\n";
		return(0);
	}

	fsTrainingImages["images"] >> matTrainingImages;
	fsTrainingImages.release();

	//KNearest kNearest = CvKNearest();
	//kNearest.train(matTrainingImages, matClassificationFloats);
	Ptr<TrainData> trainData = TrainData::create(matTrainingImages, ROW_SAMPLE, matClassificationFloats);
	Ptr<KNearest> kNearest = KNearest::create();
	kNearest->setDefaultK(1);
	kNearest->setIsClassifier(true);
	kNearest->train(trainData);

	
	cv::Mat matTestingNumbers = cv::imread("3.png");

	if (matTestingNumbers.empty()){
		std::cout << "error:image not read from file\n\n";
		return(0);
	}

	cv::Mat matGrayscale;
	cv::Mat matBlurred;
	cv::Mat matThresh;
	cv::Mat matThreshCopy;

	cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);

	cv::GaussianBlur(matGrayscale,
		matBlurred,
		cv::Size(15, 15),
		0);

	//cv::adaptiveThreshold(matBlurred,
	//	matThresh,
	//	255,
	//	cv::ADAPTIVE_THRESH_GAUSSIAN_C,
	//	cv::THRESH_BINARY_INV,
	//	11,
	//	2);
	
	cv::threshold(matBlurred, matThresh, 70, 255, cv::THRESH_BINARY);

	//threshold(matGrayscale, matThresh, 70, 255, CV_THRESH_BINARY_INV);//大于48像素为0（黑色），小于则为255（白色）
	//threshold(matGrayscale, matThresh, 70, 255, CV_THRESH_BINARY);//大于48像素为0（黑色），小于则为255（白色）
	//执行开运算，先腐蚀去除噪点，后膨胀
	/*cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::morphologyEx(matThresh, matThresh, cv::MORPH_OPEN, element);*/
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	cv::dilate(matThresh, matThresh, element);
	matThreshCopy = matThresh.clone();

	std::vector<std::vector<cv::Point>> ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	cv::findContours(matThreshCopy,
		ptContours,
		v4iHierarchy,
		cv::RETR_EXTERNAL,
		cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++){
		ContourWithData contourWithData;
		contourWithData.ptContour = ptContours[i];
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
		allContoursWithData.push_back(contourWithData);

	}

	for (int i = 0; i < allContoursWithData.size(); i++)
	{
		if (allContoursWithData[i].checkIfCotourIsValid()){
			validContoursWithData.push_back(allContoursWithData[i]);
		}
	}

	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	std::string strFinalString;

    //有几个字符，就循环几次。每一次框出字符并KNearest预测字符的值，并添加到现实的字符串中
	for (int i = 0; i < validContoursWithData.size(); i++) 
	{

		cv::rectangle(matTestingNumbers,                 //数字外面的方框
			validContoursWithData[i].boundingRect,
			cv::Scalar(255, 0, 0),
			2);

		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);

		cv::Mat matROIResized;  //ROI区域
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

		cv::Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);

		//float fltCurrentChar = kNearest.findNearest(matROIFloat.reshape(1, 1), 1);
		float fltCurrentChar = kNearest->predict(matROIFloat.reshape(1, 1));

		strFinalString = strFinalString + char(int(fltCurrentChar));

	}

	std::cout << "\n\n" << strFinalString << "\n\n";

	cv::imshow("matTestingNumbers", matTestingNumbers);

	cv::waitKey(0);

	return(0);

}