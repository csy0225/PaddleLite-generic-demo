#pragma once

#include <string>
#include <vector>
#include <map>

namespace paddle {
namespace benchmark {
    
typedef struct BenchmarkArgs {
    std::string model_name{""};
    std::string model_dir{""};
    std::string model_file{""};
    std::string param_file{""};
    std::string optimized_model_dir{""};
    std::map<std::string, std::vector<float>> inputs_shape;
    std::vector<int> batch_sizes;
    std::string dataset_dir;
    std::vector<std::string> metrics;
    std::vector<std::string> nnadapter_device_names;
    std::string nnadapter_context_properties{""};
    std::string nnadapter_subgraph_partition_config_path{""};
    int power_mode{3};
    int threads{1};
    int warmup{10};
    int repeats{100};
} BenchmarkArgs;

typedef struct LatencyInfo {
    double latency_avg_time;
    double latency_min_time;
    double latency_max_time;
} LatencyInfo;

typedef struct TopKInfo {
    double top1_accuracy;
    double top5_accuracy;
} TopKInfo;

typedef struct ThroughputInfo {
    double throughput;
} ThroughputInfo;

} // namespace benchmark
} // namespace paddle