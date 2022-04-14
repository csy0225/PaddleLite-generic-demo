#include "benchmark.hpp"
#include <iostream>

namespace paddle {
namespace benchmark {

void BenchmarkDemo::CreatePaddlePredictor() {
#ifdef USE_FULL_API
  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  if (!benchmarkArgs_.model_file.empty() && !benchmarkArgs_.param_file.empty()) {  // combined model
    cxx_config.set_model_file(benchmarkArgs_.model_file);
    cxx_config.set_param_file(benchmarkArgs_.param_file);
  } else if (!benchmarkArgs_.model_dir.empty()) {
    cxx_config.set_model_dir(benchmarkArgs_.model_dir);
  }
  cxx_config.set_threads(benchmarkArgs_.threads);
  cxx_config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(benchmarkArgs_.power_mode));
  std::vector<paddle::lite_api::Place> valid_places;
  if (std::find(benchmarkArgs_.nnadapter_device_names.begin(),
                benchmarkArgs_.nnadapter_device_names.end(),
                "cpu") == benchmarkArgs_.nnadapter_device_names.end()) {
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
    valid_places.push_back(
        paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
  }
#if defined(__arm__) || defined(__aarch64__)
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
#elif defined(__x86_64__)
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kX86), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)});
#endif
  cxx_config.set_valid_places(valid_places);
  cxx_config.set_nnadapter_device_names(benchmarkArgs_.nnadapter_device_names);
  cxx_config.set_nnadapter_context_properties(benchmarkArgs_.nnadapter_context_properties);
  // Set the subgraph custom partition configuration file
  if (!benchmarkArgs_.nnadapter_subgraph_partition_config_path.empty()) {
    std::vector<char> nnadapter_subgraph_partition_config_buffer;
    if (read_file(benchmarkArgs_.nnadapter_subgraph_partition_config_path,
                  &nnadapter_subgraph_partition_config_buffer,
                  false)) {
      if (!nnadapter_subgraph_partition_config_buffer.empty()) {
        std::string nnadapter_subgraph_partition_config_string(
            nnadapter_subgraph_partition_config_buffer.data(),
            nnadapter_subgraph_partition_config_buffer.size());
        cxx_config.set_nnadapter_subgraph_partition_config_buffer(
            nnadapter_subgraph_partition_config_string);
      }
    } else {
      throw std::logic_error(
          "Failed to load the subgraph custom partition configuration file " + benchmarkArgs_.nnadapter_subgraph_partition_config_path);
    }
  }
  predictor_ = paddle::lite_api::CreatePaddlePredictor(cxx_config);
  predictor_->SaveOptimizedModel(
        benchmarkArgs_.optimized_model_dir, paddle::lite_api::LiteModelType::kNaiveBuffer);
#endif
  // Run inference by using light api with MobileConfig
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(benchmarkArgs_.optimized_model_dir + ".nb");
  mobile_config.set_threads(benchmarkArgs_.threads);
  mobile_config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(benchmarkArgs_.power_mode));
  mobile_config.set_nnadapter_device_names(benchmarkArgs_.nnadapter_device_names);
  mobile_config.set_nnadapter_context_properties(benchmarkArgs_.nnadapter_context_properties);
  predictor_ = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
        mobile_config);
}

void BenchmarkDemo::ReadDatasetLabels() {
  // Read Labels
  std::string labels_txt = benchmarkArgs_.dataset_dir + "/labels/labels.txt";
  auto label_lines = ReadLines(labels_txt);
  for (int i = 0; i < label_lines.size(); i++) {
      std::string image_id = Split(label_lines[i], " ")[0].substr(4,28);
      int label = std::stoi(Split(label_lines[i], " ")[1]);
      labels_[i] = std::make_pair(image_id, label);
  }
}

void BenchmarkDemo::InitMetrics() {
    auto metrics = std::make_shared<Metrics>(benchmarkArgs_);
    metrics_.push_back(metrics);
    for (auto metric : benchmarkArgs_.metrics) {
        if (metric == "topk") {
            topk_metric_ = std::make_shared<TopKMetric>(benchmarkArgs_, 5);
            metrics_.push_back(std::static_pointer_cast<Metrics>(topk_metric_));
        }
        if (metric == "latency") {
            latency_metric_ = std::make_shared<LatencyMetric>(benchmarkArgs_);
            metrics_.push_back(std::static_pointer_cast<Metrics>(latency_metric_));
        }
        if (metric == "throughput") {
            throughput_metric_ = std::make_shared<ThroughputMetric>(benchmarkArgs_);
            metrics_.push_back(std::static_pointer_cast<Metrics>(throughput_metric_));
        }
    }
}

void BenchmarkDemo::ShowMetrics() {
    for (auto metric : metrics_) {
        metric->Compute();
        metric->ShowMessage();
        metric->DumpMessage();
    }
}

void BenchmarkDemo::FeedInputTensor(int batch_size, int image_idx, int total_image_num) {
    std::vector<std::string> input_names = predictor_->GetInputNames();
    int inputs_total_data_count = 0;
    for (auto input_name : input_names) {
        int input_data_count = 1;
        auto input_tensor = predictor_->GetInputByName(input_name);
        std::vector<int64_t> input_shape;
        input_shape.push_back(batch_size);
        for (int i = 1; i < benchmarkArgs_.inputs_shape[input_name].size(); i++) {
            input_shape.push_back(benchmarkArgs_.inputs_shape[input_name][i]);
            input_data_count *= benchmarkArgs_.inputs_shape[input_name][i];
        }
        input_tensor->Resize(input_shape);
        inputs_total_data_count += input_data_count;
    }

    // data rearrangement
    std::vector<std::pair<int, float*>> inputs_tensor_info; // <tensor_data_num, data_addr>
    for (int i = 0; i < input_names.size(); i++) {   
        auto input_tensor = predictor_->GetInput(i);
        auto input_tensor_shape = input_tensor->shape();
        int input_tensor_data_count = 1;
        for (int i = 0; i < input_tensor_shape.size(); i++) {
            input_tensor_data_count *= input_tensor_shape[i];
        }
        inputs_tensor_info.push_back(std::make_pair(input_tensor_data_count, input_tensor->mutable_data<float>()));
    }

    int idx = 0;
    for (int i = image_idx; i < image_idx + batch_size; i++) { 
        std::string image_id = (i >= total_image_num) ? labels_[i-total_image_num].first : labels_[i].first;
        std::string image_data_path = benchmarkArgs_.dataset_dir + "/data/" + image_id;
        const std::vector<float>& image_data = ReadInputData(image_data_path, inputs_total_data_count);
        int input_data_count = 0;
        for (int i = 0; i < input_names.size(); i++) {
            int data_num = inputs_tensor_info[i].first / batch_size;
            float* data_addr = inputs_tensor_info[i].second;
            auto tensor_data_addr_in_image = image_data.data();
            memcpy(data_addr + idx * data_num, image_data.data() + input_data_count, data_num * sizeof(float));
            input_data_count += data_num;
        }
        idx++;
    }
}

void BenchmarkDemo::CollectPrecisionMetric(int image_idx, int total_image_num) {
        std::vector<std::string> output_names = predictor_->GetOutputNames();
        for (auto output_name : output_names) {
            std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
                std::move(predictor_->GetTensor(output_name)));
            auto output_shape = output_tensor->shape();
            auto output_data = output_tensor->data<float>();
            auto batch_size = output_shape[0];
            auto output_data_count = 1;
            for (int i = 1; i < output_shape.size(); i++) {
                output_data_count *= output_shape[i];
            }
            std::vector<std::vector<float>> results(batch_size, std::vector<float>(output_data_count));
            for (int i = 0; i < batch_size; i++) {
                memcpy(&(results[i].at(0)), output_data, output_data_count * sizeof(float));
                output_data += output_data_count;
            }
            std::vector<int> image_labels;
            for (int i = image_idx; i < image_idx + batch_size; i++) {
                int label = (i >= total_image_num) ? labels_[i-total_image_num].second : labels_[i].second;
                image_labels.push_back(label);
            }
            if (topk_metric_ != nullptr) {
                topk_metric_->total_image_num = total_image_num;
                topk_metric_->Collect(results, image_labels, batch_size);
            }
        }
}

void BenchmarkDemo::CollectPerformanceMetrics(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
    double total_time_cost) {
    if (latency_metric_ != nullptr) {
        latency_metric_->Collect(batch_size, repeat_count, min_time_cost, 
            max_time_cost, total_time_cost);
    }
    if (throughput_metric_ != nullptr) {
        throughput_metric_->Collect(batch_size, repeat_count, min_time_cost, 
            max_time_cost, total_time_cost);
    }
}

void BenchmarkDemo::Test() {
    size_t iter_times = benchmarkArgs_.batch_sizes.size();
    int total_image_num = labels_.size();
    for (auto iter_idx = 0; iter_idx < iter_times; iter_idx++) {
       CreatePaddlePredictor();
       auto batch_size = benchmarkArgs_.batch_sizes[iter_idx];
       // warm up
       for (int i = 0; i < benchmarkArgs_.warmup; i++) {
           FeedInputTensor(batch_size, 0, total_image_num);
           predictor_->Run();
       }

       // test
       double max_time_cost = 0.0f;
       double min_time_cost = std::numeric_limits<float>::max();
       double total_time_cost = 0.0f;
       int inner_iter_count = 0;
       for (int image_idx = 0; image_idx < total_image_num; image_idx += batch_size) {
           FeedInputTensor(batch_size, image_idx, total_image_num);
           auto start = get_current_us();
           predictor_->Run();
           auto end = get_current_us();
           inner_iter_count++;
           double cur_time_cost = (end - start) / 1000.0f;
           if (cur_time_cost > max_time_cost) {
               max_time_cost = cur_time_cost;
           }
           if (cur_time_cost < min_time_cost) {
               min_time_cost = cur_time_cost;
           }
           total_time_cost += cur_time_cost;
           std::cout << "iter_index: " << (image_idx / batch_size)
                     << " image_id: " << "[" << image_idx << "-" << image_idx + batch_size << ")"
                     << " prediction time: " << cur_time_cost << " ms"
                     << " total prediction time: " << total_time_cost << "ms" << std::endl;
           CollectPrecisionMetric(image_idx, total_image_num);
       }
       CollectPerformanceMetrics(batch_size, inner_iter_count, min_time_cost, 
            max_time_cost, total_time_cost);    
    }
}

} // namespace benchmark
} // namespace paddle