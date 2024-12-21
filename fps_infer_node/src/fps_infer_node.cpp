#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <fstream>
#include <unistd.h>  // usleep 函数
#include <rclcpp/rclcpp.hpp>

using namespace std;
using namespace ov;
using namespace cv;

class FPSInferNode : public rclcpp::Node {
public:
    FPSInferNode() : Node("fps_infer_node") {
        // 从参数服务器获取路径
        this->declare_parameter<string>("model_file", "/home/dji/projects/zwc/fps_infer_node/models/last.onnx");
        this->declare_parameter<string>("image_file", "/home/dji/projects/zwc/fps_infer_node/models/new.jpg");
        this->declare_parameter<string>("output_file", "fps_output.txt");

        this->get_parameter("model_file", model_file_);
        this->get_parameter("image_file", image_file_);
        this->get_parameter("output_file", output_file_);

        // 初始化 OpenVINO 推理环境
        core_ = std::make_unique<Core>();
        compiled_model_ = core_->compile_model(model_file_, "GPU");
        infer_request_ = compiled_model_.create_infer_request();

        // 打开 FPS 输出文件
        fps_file_.open(output_file_, ios::out);
        if (!fps_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open output file!");
            rclcpp::shutdown();
        }

        // 定时器，每隔一段时间执行推理
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), std::bind(&FPSInferNode::infer_callback, this));
    }

private:
    void infer_callback() {
        // 读取并调整图像

        clock_t start_time = clock();
        if (frame_.empty()) {
            frame_ = imread(image_file_, IMREAD_COLOR);
            if (frame_.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load image!");
                rclcpp::shutdown();
                return;
            }
            resize(frame_, image_, Size(640, 640));
            image_.convertTo(image_, CV_32F, 1.0 / 255.0);
        }

        // 设置模型输入
        Tensor input_tensor = infer_request_.get_input_tensor();
        float* input_tensor_data = input_tensor.data<float>();
        memcpy(input_tensor_data, image_.data, sizeof(uint8_t) * 3 * 640 * 640);

        // 推理开始
        // clock_t start_time = clock();
        infer_request_.infer();
        float time_taken = (clock() - start_time) / static_cast<float>(CLOCKS_PER_SEC);
        float time_taken_ms = time_taken * 1000;

        // 计算 FPS
        float fps = 1.0 / time_taken;
        RCLCPP_INFO(this->get_logger(), "Infer time(ms): %.2f, FPS: %.2f", time_taken_ms, fps);

        // 写入 FPS 到文件
        fps_file_ << "Infer time(ms): " << time_taken_ms << ", FPS: " << fps << endl;
    }

    string model_file_;
    string image_file_;
    string output_file_;
    Mat frame_, image_;
    std::unique_ptr<Core> core_;
    CompiledModel compiled_model_;
    InferRequest infer_request_;
    ofstream fps_file_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FPSInferNode>());
    rclcpp::shutdown();
    return 0;
}
