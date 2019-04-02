// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem.hpp>
#include <time.h>
//#include <class_map.h>
#define  fs boost::filesystem
using std::cout;
using std::endl;


using namespace caffe;  // NOLINT(build/namespaces)


struct model_args{
    string model_file;
    string weights_file;
    string label_map_file;
    string mean_file;
    string mean_value;
    float confidence_threshold;
};

struct vedio_args{
    cv::Size size;
    int fps = 30;
    bool color = true;
    vector<char> CV_FOURCC = {'M', 'J', 'P', 'G'};

};


void PrintAllFile(fs::path full_path, vector<string>& file_list)
{
    if (fs::exists(full_path))
    {
        fs::directory_iterator item_begin(full_path);
        fs::directory_iterator item_end;
        for (; item_begin != item_end; item_begin++)
        {
            if (fs::is_directory(*item_begin))
            {

                cout << item_begin->path() << "\t[dir]" << endl;
                PrintAllFile(item_begin->path(), file_list);
            }
            else
            {
                file_list.push_back(item_begin->path().string());
                cout << item_begin->path() << endl;
            }
        }
    }
}





class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}



std::vector<vector<float >> get_ssd_result_once(Detector &detector, cv::Mat &img, float confidence_threshold){
    std::vector<vector<float> > detections = detector.Detect(img);
    std::vector<vector<float >> results;
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
            results.push_back(d);
        }
    }
    return results;
}
DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "video",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.2,
    "Only store detections with score higher than the threshold.");

string show_time(){
    time_t t0 = time(NULL);
    tm* t = localtime(&t0);
    string time;
    cout<<"date: "
        <<t->tm_year + 1900 << "年"
        <<t->tm_mon + 1 << "月"
        <<t->tm_mday <<"日"
        <<t->tm_hour <<"时"
        <<t->tm_min << "分"
        <<t->tm_sec<< "秒"
        <<endl;
    time += std::to_string(t->tm_year + 1900) +"-";
    time += std::to_string(t->tm_mon + 1) +"-";
    time += std::to_string(t->tm_mday) +"-";
    time += std::to_string(t->tm_hour) +"-";
    time += std::to_string(t->tm_min) +"-";
    time += std::to_string(t->tm_sec);
    return time;
};

vector<string> split(const string &s, const string &seperator){
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while(i != s.size()){
        //找到字符串中首个不等于分隔符的字母；
        int flag = 0;
        while(i != s.size() && flag == 0){
            flag = 1;
            for(string_size x = 0; x < seperator.size(); ++x)
                if(s[i] == seperator[x]){
                    ++i;
                    flag = 0;
                    break;
                }
        }

        //找到又一个分隔符，将两个分隔符之间的字符串取出；
        flag = 0;
        string_size j = i;
        while(j != s.size() && flag == 0){
            for(string_size x = 0; x < seperator.size(); ++x)
                if(s[j] == seperator[x]){
                    flag = 1;
                    break;
                }
            if(flag == 0)
                ++j;
        }
        if(i != j){
            result.push_back(s.substr(i, j-i));
            i = j;
        }
    }
    return result;
}
vector<string> get_label_map(string file){
    std::ifstream myfile(file, ios::in);
    if (!myfile.is_open())
    {
        cout << "未成功打开文件" <<file<< endl;
    }
    vector<string> label_map;
    label_map.push_back("none");
    string temp;
    int idx1, idx2;
    string name;
    while(getline(myfile,temp))
    {
//        cout << "label name: "<< temp<< endl;
        vector<string>::iterator pt = split(temp, ",").end()-1;
        name = *pt;
        label_map.push_back(name);
    }
    myfile.close();
    return  label_map;
}

void draw_one_obj(cv::Mat *img, vector<vector<float >> *results, vector<string> &label_map){

    for (int i = 0; i < results->size(); ++i) {
        const vector<float> &d = (*results)[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].

        float score = d[2];
        cv::Point2d point1(d[3] * img->cols, d[4] * img->rows);
        cv::Point2d point2(d[5] * img->cols, d[6] * img->rows);
        cv::rectangle(*img, point1, point2, cv::Scalar(255, 255, 255), 3);
        string label_name = label_map[int(d[1])];
        cv::Point2d point_txt(d[3] * img->cols, (d[4] * img->rows) - 10);
        string show_label = label_name+""+std::to_string(score).substr(0, 5);
        cv::putText(*img, show_label, point_txt, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    }
};

void process_images(Detector detector,
        string image_dir,
        string output_dir,
        vector<string> &label_map,
        float confidence_threshold){
// Process image one by one.
    vector<string> img_list;
    fs::path image_dir_path(image_dir);
    PrintAllFile(image_dir_path, img_list);
    cout<<"图片总数="<<img_list.size()<<endl;
    for (int i=0; i< img_list.size(); i++) {
        string file = img_list[i];
        cv::Mat img = cv::imread(file, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;
        vector<vector<float >> results;
        results=get_ssd_result_once(detector, img, confidence_threshold);
        /* Print the detection results. */
        draw_one_obj(&img, &results, label_map);
        cv::imshow("图片", img); //显示一帧画面
        cv::waitKey(5000); //等待键盘进行下一张
        string file_name = *(split(file, "/").end()-1);
        cv::imwrite(output_dir +"/" +file_name, img);
    }
};

void process_video(Detector detector,
        cv::VideoCapture cap,
        string output_vedio,
        vedio_args vedio_args_,
        model_args model_args_,
        vector<string> label_map){

    if (!cap.isOpened()) {
        cout << "Failed to open video: "<< endl;
        return;
    }

    cv::Mat img;
    bool success = cap.read(img);
    if (!success) {
        cout << "Failed to open video's first frame! " << endl;
        return;
    }
    cout << img.size()<< endl;
    cv::VideoWriter v_writer;
    vector<char > CV_FOURCC_ = vedio_args_.CV_FOURCC;
    v_writer.open(output_vedio, CV_FOURCC(CV_FOURCC_[0], CV_FOURCC_[1], CV_FOURCC_[2], CV_FOURCC_[3]),
            10, img.size(), vedio_args_.color);
    int frame_count = 0;
    while (true) {
        bool success = cap.read(img);
        if (!success) {
            cout << "Processed " << frame_count << endl;
            break;
        }
        CHECK(!img.empty()) << "Error when read frame";
        vector<vector<float >> results;
        results=get_ssd_result_once(detector, img, model_args_.confidence_threshold);

        /* Print the detection results. */
        draw_one_obj(&img, &results, label_map);

        cv::imshow("视频显示", img); //显示一帧画面
        cout<<img.size()<<endl;
        v_writer.write(img);
        cv::waitKey(1); //延时30ms
        ++frame_count;
        if (frame_count >100000) {
            cout << "Processed " << frame_count << endl;
            break;
        }
    }
    if (cap.isOpened()) {
        cap.release();
    }
};


int main() {
    string mode = "phone"; //images, vedio, camera, phone
    char wsdir[1000];
    getcwd(wsdir, 1000);
    std::cout<<"当前工作路径: "<<wsdir<<std::endl;
    string time_str = show_time();

    model_args VGG_coco_SSD_300x300;
    VGG_coco_SSD_300x300.model_file = "../model/coco/deploy.prototxt";
    VGG_coco_SSD_300x300.weights_file = "../model/coco/VGG_coco_SSD_300x300_iter_400000.caffemodel";
    VGG_coco_SSD_300x300.label_map_file = "../model/coco/labels.txt";
    VGG_coco_SSD_300x300.mean_file = "";
    VGG_coco_SSD_300x300.mean_value = "104,117,123";
    VGG_coco_SSD_300x300.confidence_threshold = 0.2;

    model_args VGG_VOC0712_SSD_300x300;
    VGG_VOC0712_SSD_300x300.model_file = "../model/voc/deploy.prototxt";
    VGG_VOC0712_SSD_300x300.weights_file = "../model/voc/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
    VGG_VOC0712_SSD_300x300.label_map_file = "../model/voc/coco_voc_map.txt";
    VGG_VOC0712_SSD_300x300.mean_file = "";
    VGG_VOC0712_SSD_300x300.mean_value = "104,117,123";
    VGG_VOC0712_SSD_300x300.confidence_threshold = 0.2;

    vedio_args mycamera_args;
    mycamera_args.fps = 10;
    mycamera_args.color = true;

    vedio_args wechat_video_args;
    wechat_video_args.fps = 29;
    wechat_video_args.color = true;

    vedio_args myphone_video_args;
    wechat_video_args.fps = 29;
    wechat_video_args.color = true;
    wechat_video_args.CV_FOURCC = {'D','I','V','X'};

    model_args using_model_args(VGG_coco_SSD_300x300);
    const string& image_dir = "../images";
    const string& image_save_dir = "../images_results";

    const string& vedio_dir = "../vedios/webwxgetvideo";
    string video_name = "vedio_result_offical-"+time_str;
    string output_vedio = "../vedio_results/"+video_name+".avi";


    vector<string> label_map(get_label_map(using_model_args.label_map_file));

    // Initialize the network.
    Detector detector(using_model_args.model_file,
            using_model_args.weights_file,
            using_model_args.mean_file,
            using_model_args.mean_value);


    // Process image one by one.
    if (mode == "images"){
        process_images(detector, image_dir, image_save_dir, label_map, using_model_args.confidence_threshold);
    }
    else if (mode == "vedio")
    {
        cv::VideoCapture cap(vedio_dir);
        process_video(detector,
                cap,
                output_vedio,
                wechat_video_args,
                using_model_args,
                label_map);
    }
    else if (mode == "camera"){
        cv::VideoCapture cap(0);
        process_video(detector,
                      cap,
                      output_vedio,
                      mycamera_args,
                      using_model_args,
                      label_map);
    }
    else if (mode == "phone"){
        cv::VideoCapture cap("http://admin:123@192.168.43.207:8081");
        process_video(detector,
                      cap,
                      output_vedio,
                      myphone_video_args,
                      using_model_args,
                      label_map);
    }
    else{
        cout << "model error:" << mode << endl;
    }

    return 0;
}

