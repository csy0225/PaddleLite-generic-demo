#pragma once

#include "struct.hpp"
#include "metrics.hpp"
#include "utils.hpp"
#include <paddle_api.h>

namespace paddle {
namespace benchmark {
    
class BenchmarkDemo {
public:
    explicit BenchmarkDemo(BenchmarkArgs &args) : benchmarkArgs_(args) {}

    void CreatePaddlePredictor();
    
    void Test();

    void ReadDatasetLabels();

    void InitMetrics();

    void ShowMetrics();

private:
    void CollectPerformanceMetrics(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
        double total_time_cost);
    void CollectPrecisionMetric(int image_idx, int total_image_num);
    void FeedInputTensor(int batch_size, int image_idx, int total_image_num);
    
    std::map<int, std::pair<std::string, int>> labels_;
    BenchmarkArgs benchmarkArgs_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_{nullptr};
    std::vector<std::shared_ptr<Metrics>> metrics_;
    std::shared_ptr<TopKMetric> topk_metric_; 
    std::shared_ptr<LatencyMetric> latency_metric_;
    std::shared_ptr<ThroughputMetric> throughput_metric_;
};


} 
}
