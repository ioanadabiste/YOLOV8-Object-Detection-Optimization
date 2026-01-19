#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono> // <--- AM ADAUGAT ASTA PENTRU TIMP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp> // Uneori necesar pentru functii GUI
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;
namespace fs = std::filesystem;

// ---------------- FUNCTII HELPER (Ramase identice) ----------------

float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;
    int target_h = target_size[0];
    int target_w = target_size[1];
    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }
    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;
    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);
    cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
}

torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0) return torch::empty({0}, bboxes.options().dtype(torch::kLong));
    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();
    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
    auto order_t = std::get<1>(scores.sort(true, 0, true));
    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));
    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();
    int64_t num_to_keep = 0;
    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1) continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i]; auto iy1 = y1[i]; auto ix2 = x2[i]; auto iy2 = y2[i]; auto iarea = areas[i];
        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1) continue;
            auto xx1 = std::max(ix1, x1[j]); auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]); auto yy2 = std::min(iy2, y2[j]);
            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold) suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;
    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));
    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) output.push_back(torch::zeros({0, 6 + nm}, prediction.device()));
    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]});
        auto x_split = x.split({4, nc, nm}, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});
        int n = x.size(0);
        if (!n) { continue; }
        auto c = x.index({Slice(), Slice{5, 6}}) * 7680;
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
        auto scores = x.index({Slice(), 4});
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({Slice(None, max_det)});
        output[xi] = x.index({i});
    }
    return torch::stack(output);
}

torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}

float box_iou(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
    float xA = std::max(boxA.x, boxB.x);
    float yA = std::max(boxA.y, boxB.y);
    float xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    float yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    float interArea = std::max(0.f, xB - xA) * std::max(0.f, yB - yA);
    float boxAArea = boxA.width * boxA.height;
    float boxBArea = boxB.width * boxB.height;

    float iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6); 
    return iou;
}

// ---------------- MAIN ----------------

int main() {
    // Alege device-ul automat (GPU daca ai drivere NVIDIA, altfel CPU)
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Rulare pe device: " << (torch::cuda::is_available() ? "CUDA (GPU)" : "CPU") << std::endl;

    std::vector<std::string> coco_classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                                      "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                      "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    std::vector<std::string> custom_classes{ "0","135","180","225","270","315","45","90" };

    try {
        // --- !!! ATENTIE: VERIFICA ACESTE CAI INAINTE DE RULARE !!! ---
        std::string yolo_path = "D:/proiectSSCv5/my_model/torchlibV2/models/yolov8n.torchscript";
        std::string my_model_path = "D:/proiectSSCv5/my_model/torchlibV2/models/my_model.torchscript";
        std::string input_folder_path = "D:/proiectSSCv5/my_model/torchlib/images/";
        std::string output_folder_path = "D:/proiectSSCv5/my_model/torchlibV2/results/";

        if (!fs::exists(output_folder_path)) {
            fs::create_directories(output_folder_path);
        }

        std::cout << "Se incarca modelele..." << std::endl;
        torch::jit::script::Module yolo_model;
        yolo_model = torch::jit::load(yolo_path, device);
        yolo_model.eval();
        // yolo_model.to(device, torch::kFloat32); // De obicei redundant daca s-a incarcat cu device, dar ok

        torch::jit::script::Module custom_model;
        custom_model = torch::jit::load(my_model_path, device);
        custom_model.eval();
        // custom_model.to(device, torch::kFloat32);
        std::cout << "Modele incarcate cu succes." << std::endl;

        // Variabile pentru statistici de timp
        double total_inference_time_ms = 0;
        int processed_frames_count = 0;

        for (const auto& entry : fs::directory_iterator(input_folder_path)) {
            if (entry.is_directory()) continue;

            std::string file_path = entry.path().string();
            std::string file_name = entry.path().filename().string();
            
            // 1. Citire imagine (NU masuram asta pentru performanta AI-ului)
            cv::Mat image = cv::imread(file_path);
            if(image.empty()) continue;

            // --- START CRONOMETRU ---
            // Masuram: Preprocesare + Inferenta Model 1 + Inferenta Model 2 + Postprocesare (NMS) + Matching
            auto start_time = std::chrono::high_resolution_clock::now();

            cv::Mat input_image;
            letterbox(image, input_image, {640, 640});
            cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

            torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte).to(device);
            image_tensor = image_tensor.toType(torch::kFloat32).div(255);
            image_tensor = image_tensor.permute({2, 0, 1});
            image_tensor = image_tensor.unsqueeze(0);
            std::vector<torch::jit::IValue> inputs {image_tensor};

            // Inferenta
            torch::Tensor output = yolo_model.forward(inputs).toTensor().cpu();
            torch::Tensor output_custom = custom_model.forward(inputs).toTensor().cpu();

            // Post-procesare (NMS)
            auto keep = non_max_suppression(output)[0];
            auto boxes = keep.index({Slice(), Slice(None, 4)});
            keep.index_put_({Slice(), Slice(None, 4)}, scale_boxes({input_image.rows, input_image.cols}, boxes, {image.rows, image.cols}));
           
            auto keep_custom = non_max_suppression(output_custom)[0];
            auto boxes_custom = keep_custom.index({Slice(), Slice(None, 4)});
            keep_custom.index_put_({Slice(), Slice(None, 4)}, scale_boxes({input_image.rows, input_image.cols}, boxes_custom, {image.rows, image.cols}));

            // Matching logic & Drawing prep
            // (Includeam asta in timp deoarece face parte din algoritmul tau de detectie custom)
            for (int i = 0; i < keep.size(0); i++) {
                float x1 = keep[i][0].item().toFloat();
                float y1 = keep[i][1].item().toFloat();
                float x2 = keep[i][2].item().toFloat();
                float y2 = keep[i][3].item().toFloat();
                // ... extragere date ...
                int cls = keep[i][5].item().toInt();

                cv::Rect2f rectYolo(x1, y1, x2 - x1, y2 - y1);
                std::string best_angle = "";
                float max_iou = 0.0f;

                for (int j = 0; j < keep_custom.size(0); j++) {
                    float cx1 = keep_custom[j][0].item().toFloat();
                    float cy1 = keep_custom[j][1].item().toFloat();
                    float cx2 = keep_custom[j][2].item().toFloat();
                    float cy2 = keep_custom[j][3].item().toFloat();
                    cv::Rect2f rectCustom(cx1, cy1, cx2 - cx1, cy2 - cy1);

                    float iou = box_iou(rectYolo, rectCustom);
                    if (iou > 0.3f && iou > max_iou) {
                        max_iou = iou;
                        int custom_cls = keep_custom[j][5].item().toInt();
                        if (custom_cls < custom_classes.size()) {
                            best_angle = custom_classes[custom_cls];
                        }
                    }
                }
                
                // Desenare (o facem aici ca sa fie gata imaginea, dar e neglijabila la timp pe CPU puternic)
                std::string yolo_cls_name = (cls < coco_classes.size()) ? coco_classes[cls] : "Obj";
                std::string label = yolo_cls_name;
                if (!best_angle.empty()) label += " | " + best_angle + "deg";
                
                cv::Scalar color = (!best_angle.empty()) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
                cv::putText(image, label, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }

            // --- STOP CRONOMETRU ---
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Calcul durata
            std::chrono::duration<double, std::milli> duration = end_time - start_time;
            double ms = duration.count();
            double fps = 1000.0 / ms;

            std::cout << "Img: " << file_name << " | Timp: " << ms << " ms | FPS: " << fps << std::endl;

            // Adunam la total (Ignoram prima imagine pentru medie, este mereu 'warm-up')
            if (processed_frames_count > 0) {
                total_inference_time_ms += ms;
            }
            processed_frames_count++;

            // Salvare imagine (NU intra la timpul de procesare)
            std::string output_path = output_folder_path + file_name;
            cv::imwrite(output_path, image);
        }

        // Afisare medie finala
        if (processed_frames_count > 1) {
            double avg_time = total_inference_time_ms / (processed_frames_count - 1);
            double avg_fps = 1000.0 / avg_time;
            std::cout << "\n============================================\n";
            std::cout << "STATISTICI FINALE (fara prima imagine):" << std::endl;
            std::cout << "Timp Mediu: " << avg_time << " ms" << std::endl;
            std::cout << "FPS Mediu:  " << avg_fps << std::endl;
            std::cout << "============================================\n";
        }
        
    } catch (const c10::Error& e) {
        std::cout << "Eroare LibTorch: " << e.msg() << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "Eroare OpenCV: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Eroare Standard: " << e.what() << std::endl;
    }

    return 0;
}