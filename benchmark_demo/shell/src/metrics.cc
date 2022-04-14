#include "metrics.hpp"
#include "utils.hpp"
#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>

namespace paddle {
namespace benchmark{

std::string Metrics::Message() {
  std::stringstream ss;
  ss << "Benchmark configuration information:" << std::endl;
  ss << "\t- model_name: " << benchmarkArgs_.model_name << std::endl;
  ss << "\t- model_dir: " << benchmarkArgs_.model_dir << std::endl;
  ss << "\t- model_file: " << benchmarkArgs_.model_file << std::endl;
  ss << "\t- param_file: " << benchmarkArgs_.param_file << std::endl;
  ss << "\t- optimized_model_dir: " << benchmarkArgs_.optimized_model_dir << std::endl;
  ss << "\t- inputs_shape: " << std::endl;
  for (auto input_shape : benchmarkArgs_.inputs_shape) {
    ss << "\t\t- input_name: " << input_shape.first << "\tinput_shape: " << Vec2Str(input_shape.second) << std::endl;
  }
  ss << "\t- batch_sizes: " << Vec2Str(benchmarkArgs_.batch_sizes) << std::endl;
  ss << "\t- dataset_dir: " << benchmarkArgs_.dataset_dir << std::endl;
  ss << "\t- metrics: " << Vec2Str(benchmarkArgs_.metrics) << std::endl;
  ss << "\t- nnadapter_device_names: " << Vec2Str(benchmarkArgs_.nnadapter_device_names) << std::endl;
  ss << "\t- nnadapter_context_properties: " << benchmarkArgs_.nnadapter_context_properties << std::endl;
  ss << "\t- nnadapter_subgraph_partition_config_path: " << benchmarkArgs_.nnadapter_subgraph_partition_config_path << std::endl;
  ss << std::endl;
  return ss.str();
}

void Metrics::ShowMessage() {
  std::cout << Message();
}

void Metrics::DumpMessage() {
  std::string filename = "output/" + benchmarkArgs_.model_name + ".txt";
  std::ofstream outfile(filename, std::ios::app);
  outfile << Message();
  outfile.close();
}

void TopKMetric::Collect(const std::vector<std::vector<float>>& results,
                 const std::vector<int>& labels,
                 int batch_size) {
  correctly_samples_num_[batch_size].resize(k_);
  for (size_t i = 0; i < results.size(); i++) {
    auto& out = results[i];
    auto cmp = [](const std::pair<int, float> a, const std::pair<int, float> b) { return a.second < b.second; };
    std::priority_queue<std::pair<int, float>,
                        std::vector<std::pair<int, float>>,
                        decltype(cmp)> out_queue(cmp);
    for (size_t j = 0; j < out.size(); j++) {
      out_queue.push(std::make_pair(static_cast<int>(j), out[j]));
    }
    for (int j = 0; j < k_; j++) {
      auto tmp = out_queue.top();
      out_queue.pop();
      if (tmp.first == labels[i]) {
        for (int k_idx = j; k_idx < k_; k_idx++) {
            correctly_samples_num_[batch_size][k_idx]++;
        }  
        continue;
      }
    }
  }
}

void TopKMetric::Compute() {
  for (auto batch_record_info: correctly_samples_num_) {
      int batch_num = batch_record_info.first;
      auto correctly_count = batch_record_info.second;
      TopKInfo topk_info;
      topk_info.top1_accuracy = correctly_count[0] * 1.0 / total_image_num;
      topk_info.top5_accuracy = correctly_count[4] * 1.0 / total_image_num;
      results_[batch_num] = topk_info;
  }
}

std::string TopKMetric::Message() {
  std::stringstream ss;
  ss << "- TopK Metric:" << std::endl;
  for (auto result : results_) {
    ss << "\t- batch_size: " << result.first << std::endl;
    ss << "\t\t- Top1 Accuracy: " << result.second.top1_accuracy << std::endl;
    ss << "\t\t- Top5 Accuracy: " << result.second.top5_accuracy << std::endl;
  }
  ss << std::endl;
  return ss.str();
}

void LatencyMetric::Collect(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
    double total_time_cost) {
    Latency latency = {
        .repeat_count = repeat_count,
        .min_time_cost = min_time_cost,
        .max_time_cost = max_time_cost,
        .total_time_cost = total_time_cost,
        .avg_time_cost = total_time_cost / repeat_count
    };
    latency_results_[batch_size] = latency;
}

void LatencyMetric::Compute() {
    for (auto batch_record_info: latency_results_) {
      int batch_size = batch_record_info.first;
      LatencyInfo latencyInfo;
      latencyInfo.latency_avg_time = batch_record_info.second.total_time_cost / batch_record_info.second.repeat_count;
      latencyInfo.latency_min_time = batch_record_info.second.min_time_cost;
      latencyInfo.latency_max_time = batch_record_info.second.max_time_cost;
      results_[batch_size] = latencyInfo;
    }
}

std::string LatencyMetric::Message() {
  std::stringstream ss;
  ss << "- Latency Metric:" << std::endl;
  for (auto result : results_) {
    ss << "\t- batch_size: " << result.first << std::endl;
    ss << "\t\t- latency_avg_time: " << result.second.latency_avg_time << std::endl;
    ss << "\t\t- latency_min_time: " << result.second.latency_min_time << std::endl;
    ss << "\t\t- latency_max_time: " << result.second.latency_max_time << std::endl;
  }
  ss << std::endl;
  return ss.str();
}


void ThroughputMetric::Collect(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
    double total_time_cost) {
    auto throughput = batch_size * 1000 / (total_time_cost / repeat_count);
    ThroughputInfo throughputInfo = {
      .throughput = throughput
    };
    results_[batch_size] = throughputInfo;
}

std::string ThroughputMetric::Message() {
  std::stringstream ss;
  ss << "- Throughput Metric:" << std::endl;
  for (auto result : results_) {
    ss << "\t- batch_size: " << result.first << std::endl;
    ss << "\t\t- throughput: " << result.second.throughput << std::endl;
  }
  ss << std::endl;
  return ss.str();
}

} // namespace benchmark
} // namespace paddle