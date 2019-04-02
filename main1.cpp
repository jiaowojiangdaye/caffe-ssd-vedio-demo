#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <time.h>
using std::cin;
using std::cout;
using std::endl;


void cv_test(){
    cv::Mat image;
    image=cv::imread("/home/aaron/mydisk/program/test/1.jpg");
    cv::namedWindow("/home/aaron/mydisk/program/test/1.jpg");
    cv::imshow("/home/aaron/mydisk/program/test/1.jpg",image);
    cv::waitKey(10000);

}


void show_time(tm *t){

    cout<<"date: "
            <<t->tm_year + 1900 << "年"
            <<t->tm_mon + 1 << "月"
            <<t->tm_mday <<"日"
            <<t->tm_hour <<"时"
            <<t->tm_min << "分"
            <<t->tm_sec<< "秒"
            <<endl;
}

void show_camera(){
    cout << "摄像头开启"<< endl;

    time_t tt1 = time(NULL);
    time_t tt2 = time(NULL);
    tm* t1= localtime(&tt1);
    tm* t2= localtime(&tt1);
    show_time(t1);
    cv::VideoCapture cap(0);
    bool run_flag = true;
    while (run_flag){
        cv::Mat frame; //定义Mat变量，用来存储每一帧
        cap>>frame; //读取当前帧方法一
        //cap.read(frame); //读取当前帧方法二
        cv::imshow("视频显示", frame); //显示一帧画面
        cv::waitKey(100); //延时30ms

        time_t tt2 = time(NULL);
        t2= localtime(&tt2);
        show_time(t2);
        if ((tt2-tt1)>3000) break;
    }

    cout<<"结束"<<endl;
}



int main(int argc,char *argv[]) {
    show_camera();
    return 0;
}
