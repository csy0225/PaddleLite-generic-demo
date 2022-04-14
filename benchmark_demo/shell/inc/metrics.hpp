#pragma once

#include "struct.hpp"
#include <vector>
#include <map>

namespace paddle {
namespace benchmark {

class Metrics {
public:
    Metrics() = default;
    
    explicit Metrics(BenchmarkArgs benchmarkArgs) : benchmarkArgs_(benchmarkArgs) {}

    virtual ~Metrics() {};

    virtual std::string Message();

    void ShowMessage();

    void DumpMessage();

    virtual void Compute() {};
public:
    BenchmarkArgs benchmarkArgs_;
};

class TopKMetric : public Metrics {
public:   
    TopKMetric() = default;
    
    TopKMetric(BenchmarkArgs benchmarkArgs, int k) : Metrics(benchmarkArgs), k_(k) {}
    
    // Collect data from datasets
    void Collect(const std::vector<std::vector<float>>& results,
                    const std::vector<int>& labels,
                    int batch_size);
    virtual std::string Message() override;
    void Compute();
public:
    int k_;
    int total_image_num;
    std::map<int, std::vector<int>> correctly_samples_num_;
    std::map<int, TopKInfo> results_;
};

typedef struct Latency {
    int repeat_count;
    double min_time_cost;
    double max_time_cost;
    double total_time_cost;
    double avg_time_cost;
} Latency;

class LatencyMetric : public Metrics {
public:
    LatencyMetric() = default;

    explicit LatencyMetric(BenchmarkArgs benchmarkArgs) : Metrics(benchmarkArgs) {}

    // Collect data from datasets
    void Collect(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
        double total_time_cost);
    // Compute the metric
    void Compute();
    virtual std::string Message() override;
public:
    std::map<int, Latency> latency_results_;
    std::map<int, LatencyInfo> results_;
};

class ThroughputMetric : public Metrics {
public:
    ThroughputMetric() = default;

    explicit ThroughputMetric(BenchmarkArgs benchmarkArgs) : Metrics(benchmarkArgs) {}

    void Collect(int batch_size, int repeat_count, double min_time_cost, double max_time_cost,
        double total_time_cost);

    // Compute the metric
    void Compute() {};

    virtual std::string Message() override;

public:
    std::map<int, ThroughputInfo> results_;
};


} // namespace benchmark
} // namespace paddle