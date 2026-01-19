#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace fs = std::filesystem;
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const { if (obj) obj->destroy(); }
};

inline float compute_iou(const cv::Rect& boxA, const cv::Rect& boxB) {
    const cv::Rect inter = boxA & boxB;
    const float areaInter = static_cast<float>(inter.area());
    const float areaUnion = static_cast<float>(boxA.area() + boxB.area() - areaInter);
    return (areaUnion > 0.0f) ? (areaInter / areaUnion) : 0.0f;
}

class YoloTRT {
public:
    explicit YoloTRT(const std::string& enginePath) {
        loadEngine(enginePath);
    }

    ~YoloTRT() {
        if (stream) cudaStreamDestroy(stream);
        for (void* ptr : buffers) if (ptr) cudaFree(ptr);
    }

    std::vector<Detection> detect(const cv::Mat& img, float conf_thres, float iou_thres, int num_classes) {
        cv::Mat blob = preprocess_img(img, inputDims.width, inputDims.height);
        cudaMemcpyAsync(buffers[inputIndex], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
        }

        if (!context->enqueueV3(stream)) return {};

    
        std::vector<float> cpu_output(outputSize / sizeof(float));
        cudaMemcpyAsync(cpu_output.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        return postprocess(cpu_output, img.size(), conf_thres, iou_thres, num_classes);
    }

private:
    std::shared_ptr<IRuntime> runtime;
    std::shared_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext, TRTDeleter> context;

    cudaStream_t stream = nullptr;
    std::vector<void*> buffers;
    int inputIndex = -1, outputIndex = -1;
    size_t inputSize = 0, outputSize = 0;
    cv::Size inputDims{ 640, 640 };

    void loadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) throw std::runtime_error("Engine not found: " + path);

        const size_t size = file.tellg();
        std::vector<char> modelData(size);
        file.seekg(0, std::ios::beg);
        file.read(modelData.data(), size);
        file.close();

        runtime.reset(createInferRuntime(gLogger), TRTDeleter());
        engine.reset(runtime->deserializeCudaEngine(modelData.data(), size), TRTDeleter());
        context.reset(engine->createExecutionContext());

        cudaStreamCreate(&stream);

        const int nbIOTensors = engine->getNbIOTensors();
        buffers.resize(nbIOTensors);

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            const Dims dims = engine->getTensorShape(name);

            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                vol *= (dims.d[j] == -1) ? (i == 0 ? 640 : 1) : dims.d[j];
            }

            // Corecție pentru dimensiuni YOLO standard dacă vol este prea mic
            if (std::string(name).find("output") != std::string::npos && vol < 8400) vol = 840000;

            const size_t sizeBytes = vol * sizeof(float);
            cudaMalloc(&buffers[i], sizeBytes);

            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                inputIndex = i;
                inputSize = sizeBytes;
                if (dims.nbDims >= 4) { inputDims = { (int)dims.d[3], (int)dims.d[2] }; }
            }
            else {
                outputIndex = i;
                outputSize = sizeBytes;
            }
        }
    }

    cv::Mat preprocess_img(const cv::Mat& img, int target_w, int target_h) {
        const float scale = std::min(static_cast<float>(target_w) / img.cols, static_cast<float>(target_h) / img.rows);
        const int new_w = static_cast<int>(img.cols * scale);
        const int new_h = static_cast<int>(img.rows * scale);

        cv::Mat resized;
        cv::resize(img, resized, { new_w, new_h });

        cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
        const int top = (target_h - new_h) / 2;
        const int left = (target_w - new_w) / 2;
        resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

        cv::Mat blob;
        cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
        return blob;
    }

    std::vector<Detection> postprocess(const std::vector<float>& output, cv::Size originalSize, float conf_thres, float iou_thres, int nc) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        constexpr int num_anchors = 8400;
        const float scale = std::min(static_cast<float>(inputDims.width) / originalSize.width, static_cast<float>(inputDims.height) / originalSize.height);
        const float pad_x = (inputDims.width - originalSize.width * scale) / 2.0f;
        const float pad_y = (inputDims.height - originalSize.height * scale) / 2.0f;

        for (int i = 0; i < num_anchors; i++) {
            float max_score = -1.0f;
            int max_class_id = -1;

            for (int c = 0; c < nc; c++) {
                const float score = output[(4 + c) * num_anchors + i];
                if (score > max_score) { max_score = score; max_class_id = c; }
            }

            if (max_score >= conf_thres) {
                const float cx = output[0 * num_anchors + i];
                const float cy = output[1 * num_anchors + i];
                const float w = output[2 * num_anchors + i];
                const float h = output[3 * num_anchors + i];

                const int left = static_cast<int>((cx - w / 2.0f - pad_x) / scale);
                const int top = static_cast<int>((cy - h / 2.0f - pad_y) / scale);

                boxes.emplace_back(left, top, static_cast<int>(w / scale), static_cast<int>(h / scale));
                scores.push_back(max_score);
                class_ids.push_back(max_class_id);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, indices);

        std::vector<Detection> results;
        for (const int idx : indices) {
            // Constrângere box în limitele imaginii
            cv::Rect b = boxes[idx] & cv::Rect(0, 0, originalSize.width, originalSize.height);
            results.push_back({ class_ids[idx], scores[idx], b });
        }
        return results;
    }
};

int main() {
    const std::string input_folder = "images";
    const std::string output_folder = "results";

    const std::vector<std::string> obj_classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    const std::vector<std::string> ang_labels = { "0", "135", "180", "225", "270", "315", "45", "90" };
    const std::vector<float> ang_values = { 0.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f, 45.0f, 90.0f };

    try {
        if (!fs::exists(output_folder)) fs::create_directory(output_folder);

        YoloTRT detector("yolov8n.engine");
        YoloTRT angle_detector("my_model.engine");

        std::cout << "Modele incarcate. Procesare folder: " << input_folder << std::endl;

        double total_ms = 0.0;
        int measured = 0, count = 0;

        for (const auto& entry : fs::directory_iterator(input_folder)) {
            const std::string path = entry.path().string();
            const std::string filename = entry.path().filename().string();

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".png" && ext != ".jpeg" && ext != ".bmp") continue;

            cv::Mat img = cv::imread(path);
            if (img.empty()) continue;

            const auto t1 = std::chrono::high_resolution_clock::now();

            auto det_results = detector.detect(img, 0.25f, 0.45f, 80);
            auto ang_results = angle_detector.detect(img, 0.25f, 0.45f, 8);

            const auto t2 = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            total_ms += ms; measured++;
            std::cout << "[" << ++count << "] " << filename << " | FPS: " << std::fixed << std::setprecision(2) << 1000.0 / ms << " | Avg: " << 1000.0 / (total_ms / measured) << std::endl;

            for (const auto& obj : det_results) {
                // Filtru pentru vehicule (car, motorcycle, bus, truck)
                if (obj.class_id != 2 && obj.class_id != 3 && obj.class_id != 5 && obj.class_id != 7) continue;

                std::string label = obj_classes[obj.class_id];
                int best_ang_idx = -1;
                float max_iou = 0.0f;

                for (const auto& ang : ang_results) {
                    float iou = compute_iou(obj.box, ang.box);
                    if (iou > 0.3f && iou > max_iou) {
                        max_iou = iou;
                        best_ang_idx = ang.class_id;
                    }
                }

                if (best_ang_idx != -1) label += " (" + ang_labels[best_ang_idx] + ")";

                cv::rectangle(img, obj.box, { 0, 0, 255 }, 2);
                cv::putText(img, label, { obj.box.x, obj.box.y - 5 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0, 0, 255 }, 2);
            }

            cv::imwrite(output_folder + "/" + filename, img);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "CRASH: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}