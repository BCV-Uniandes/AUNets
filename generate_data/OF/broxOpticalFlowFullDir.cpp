#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void getFlowField(const Mat& u, const Mat& v, Mat& flowField);
void getFlowFieldAsIwant(const Mat& u, const Mat& v, Mat& flowField);
void getFlowFieldMod(const Mat& u, const Mat& v, Mat& flowField);

//FileSystem and sor utils
bool numeric_string_compare(const std::string& s1, const std::string& s2);
bool is_not_digit(char c);
vector<string> getFileNamesFromDirAsVector(const char* path);


int main(int argc, const char* argv[]){
    try{
        const char* keys =
           "{ h  | help      | false | print help message }"
           "{ s  | source    |       | specify input dir }"
           "{ z  | ziel      |       | specify target dir }"
           "{ s  | scale     | 0.8   | set pyramid scale factor }"
           "{ a  | alpha     | 0.197 | set alpha }"
           "{ g  | gamma     | 50.0  | set gamma }"
           "{ i  | inner     | 10    | set number of inner iterations }"
           "{ o  | outer     | 77    | set number of outer iterations }"
           "{ si | solver    | 10    | set number of basic solver iterations }"
           "{ t  | time_step | 0.1   | set frame interpolation time step }";

        CommandLineParser cmd(argc, argv, keys);

        if (cmd.get<bool>("help")){
            cout << "Usage: brox_optical_flow [options]" << endl;
            cout << "Avaible options:" << endl;
            cmd.printParams();
            return 0;
        }

        //use GPU 0
		setDevice(0);

        //source and target dirs
        string sourceDir = cmd.get<string>("source");
        string targetDir = cmd.get<string>("ziel");

        float scale = cmd.get<float>("scale");
        float alpha = cmd.get<float>("alpha");
        float gamma = cmd.get<float>("gamma");
        int inner_iterations = cmd.get<int>("inner");
        int outer_iterations = cmd.get<int>("outer");
        int solver_iterations = cmd.get<int>("solver");
        float timeStep = cmd.get<float>("time_step");

        if (sourceDir.empty() || targetDir.empty()){
            cerr << "Missing source/target direcories" << endl;
            return -1;
        }
        
        vector<string> filesInDir=getFileNamesFromDirAsVector(sourceDir.c_str());
        BroxOpticalFlow d_flow(alpha, gamma, scale, inner_iterations, outer_iterations, solver_iterations);
        
        cout<<"Calculate flow for dir "<<sourceDir.c_str()<<endl;
        for (vector<string>::size_type i = 0; i <(filesInDir.size()-1); ++i){
			
			//ignore these fuckers
			if (filesInDir[i].compare(".") == 0 || filesInDir[i].compare("..") == 0){
					continue;
			}

			clock_t fullTimBegin = clock();
			string frame0Name=sourceDir+"/"+filesInDir[i] ;
			string frame1Name=sourceDir+"/"+filesInDir[i+1];
			
			//cout<<"frame0Name "<<frame0Name<<endl;
			//cout<<"frame1Name "<<frame1Name<<endl;
					
			clock_t beginLoad = clock();
				Mat frame0Color = imread(frame0Name);
				Mat frame1Color = imread(frame1Name);
			clock_t endLoad = clock();
			double loadTimeSecs = double(endLoad - beginLoad) / CLOCKS_PER_SEC;
			//cout<<"Image Load Time "<<loadTimeSecs<<endl;

			if (frame0Color.empty() || frame1Color.empty()){
				cout << "Can't load input images" << endl;
				return -1;
			}
					
			clock_t beginImageTransforms = clock();
				frame0Color.convertTo(frame0Color, CV_32F, 1.0 / 255.0);
				frame1Color.convertTo(frame1Color, CV_32F, 1.0 / 255.0);
			   
				Mat frame0Gray, frame1Gray;
				cvtColor(frame0Color, frame0Gray, COLOR_BGR2GRAY);
				cvtColor(frame1Color, frame1Gray, COLOR_BGR2GRAY);

				GpuMat d_frame0(frame0Gray);
				GpuMat d_frame1(frame1Gray);
				
			clock_t endImageTransforms  = clock();
			double transformTimeSecs = double(endImageTransforms - beginImageTransforms) / CLOCKS_PER_SEC;
			//cout<<"Transform Time "<<transformTimeSecs<<endl;

			GpuMat d_fu, d_fv;
			
			clock_t beginActual = clock();
				d_flow(d_frame0, d_frame1, d_fu, d_fv);
			clock_t endActual = clock();
			double timeActual = double(endActual - beginActual) / CLOCKS_PER_SEC;
			//cout<<"Actual Calculation Time "<<timeActual<<endl;
				   
			clock_t myTransformBegin = clock();
				Mat flowFieldAsIWant;
				getFlowFieldMod(Mat(d_fu), Mat(d_fv), flowFieldAsIWant);
			clock_t myTransformEnd = clock();
			double timeMyTransform = double(myTransformEnd - myTransformBegin) / CLOCKS_PER_SEC;
			//cout<<"My Transform Time "<<timeMyTransform<<endl;

			imwrite(targetDir+"/"+filesInDir[i], flowFieldAsIWant );
			
			clock_t fullTimeEnd = clock();
			double timeFull= double(fullTimeEnd - fullTimBegin) / CLOCKS_PER_SEC;
			//cout<<" Time Full"<<timeFull<<endl;
				
		}
    }
    catch (const exception& ex){
        cerr << ex.what() << endl;
        return -1;
    }
    catch (...){
        cerr << "Unknow error" << endl;
        return -1;
    }
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}

void getFlowFieldMod(const Mat& u, const Mat& v, Mat& flowField)
{

    //Mat u=uu-mean(uu);
    //Mat v=vv-mean(vv);
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC3);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec3b* row = flowField.ptr<Vec3b>(i);

        for (int j = 0; j < flowField.cols; ++j){
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
        }
    }
}

void getFlowFieldAsIwant(const Mat& u, const Mat& v, Mat& flowField)
{
  
   Mat uMeanSubtracted=u-mean(u);
   normalize(uMeanSubtracted, uMeanSubtracted, 0.0f, 255.0f, NORM_MINMAX);
   //normalize(u, u, 0.0f, 255.0f, NORM_MINMAX);
  
   Mat vMeanSubtracted=v-mean(v);
   normalize(vMeanSubtracted, vMeanSubtracted, 0.0f, 255.0f, NORM_MINMAX);
   //normalize(v, v, 0.0f, 255.0f, NORM_MINMAX);
     
   flowField.create(u.size(), CV_8UC3);
   
   for (int i = 0; i < flowField.rows; ++i){
        const float* ptr_u = uMeanSubtracted.ptr<float>(i);
        const float* ptr_v = vMeanSubtracted.ptr<float>(i);

        Vec3b* row = flowField.ptr<Vec3b>(i);

        Point2f zeroPoint(0,0);
        for (int j = 0; j < flowField.cols; ++j){
            row[j][0] = static_cast<unsigned char> (ptr_v[j]);
            row[j][1] = static_cast<unsigned char> (ptr_u[j]);
            
            Point2f vectFlow(row[j][2],row[j][1]);
            row[j][2] = static_cast<unsigned char> (cv::norm(cv::Mat(zeroPoint),cv::Mat(vectFlow)));
            //row[j][0] = static_cast<unsigned char> (0);
        }
    }
    
    //Size size(256,256);//the dst image size,e.g.100x100
    //resize(flowField,flowField,size);//resize image
}


/*
void getFlowFieldAsIwant(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC3);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec3b* row = flowField.ptr<Vec3b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            
            
            row[j][0] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][1] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( (-ptr_v[j]*-ptr_v[j])+ (ptr_u[j]*ptr_u[j]), -maxDisplacement, maxDisplacement, 0.0f, 255.0f));;
        }
    }
}*/


vector<string> getFileNamesFromDirAsVector(const char* path){
	DIR *dir;
	struct dirent *ent;
		
	vector<string> imageNames;
	
	if ((dir = opendir (path)) != NULL) {

		while ((ent = readdir (dir)) != NULL) {
			imageNames.push_back(ent->d_name);
		}
		closedir (dir);
	} else {
		cout<<"could not open directory"<<endl;
		return imageNames;
	}
	
	//this sor workd only for stringing composed of numbers and extension, fails otherwise
	std::sort(imageNames.begin(), imageNames.end(), numeric_string_compare);

    /*for (vector<string>::size_type i = 0; i != imageNames.size(); ++i)
        cout << imageNames[i] << endl;*/

	return imageNames;
}


bool is_not_digit(char c)
{
    return !std::isdigit(c);
}

bool numeric_string_compare(const std::string& s1, const std::string& s2)
{
    // handle empty strings...

    std::string::const_iterator it1 = s1.begin(), it2 = s2.begin();

    if (std::isdigit(s1[0]) && std::isdigit(s2[0])) {
        int n1, n2;
        std::stringstream ss(s1);
        ss >> n1;
        ss.clear();
        ss.str(s2);
        ss >> n2;

        if (n1 != n2) return n1 < n2;

        it1 = std::find_if(s1.begin(), s1.end(), is_not_digit);
        it2 = std::find_if(s2.begin(), s2.end(), is_not_digit);
    }

    return std::lexicographical_compare(it1, s1.end(), it2, s2.end());
}
