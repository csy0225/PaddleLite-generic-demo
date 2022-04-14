

#include "benchmark.hpp"
#include <iostream>
#include <gflags/gflags.h>

using namespace paddle::benchmark;

DEFINE_string(model_name,
            "",
            "Model name. Set it in order to generate the benchmark result text.");
DEFINE_string(model_dir,
            "",
            "Model dir path. Set it when the model is uncombined format.");
DEFINE_string(model_file,
            "",
            "Model file. Set it when the model is combined format.");
DEFINE_string(param_file,
            "",
            "Param file. Set it when the model is combined format.");
DEFINE_string(optimized_model_dir, "", "Optimized model dir.");
DEFINE_int32(power_mode,
            3,
            "power mode: "
            "0 for POWER_HIGH;"
            "1 for POWER_LOW;"
            "2 for POWER_FULL;"
            "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 10, "warmup times");
DEFINE_int32(repeats, 100, "repeats times");
DEFINE_string(inputs_shape,
             "",
             "Inputs shape.");
DEFINE_string(nnadapter_device_names,
            "",
            "NNAdapter device names.");
DEFINE_string(nnadapter_context_properties,
            "",
            "NNAdapter context properties.");
DEFINE_string(nnadapter_subgraph_partition_config_path,
            "",
            "NNAdapter subgraph partition config path.");
DEFINE_string(batch_sizes,
            "",
            "Batch size list. Please use ',' as delimiter");
DEFINE_string(dataset_dir,
            "",
            "Dataset path.");
DEFINE_string(metrics,
            "",
            "Metrics.");

bool CheckAndParserArgs(int argc, char* argv[], BenchmarkArgs& benchmarkArgs) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    // Parser args
    if (FLAGS_model_name.empty()) {
        throw std::logic_error("Model name is required but not set. Please set \
            only --model_name option.");
    }
    benchmarkArgs.model_name = FLAGS_model_name;

    if (FLAGS_model_dir.empty() &&
        (FLAGS_model_file.empty() || FLAGS_param_file.empty()) ||
        FLAGS_optimized_model_dir.empty()) {
        // show_usage();
        throw std::logic_error("Model is required but not set. Please set \
            only --model_dir or both --model_file and --param_file option.");
    }
    benchmarkArgs.model_dir = FLAGS_model_dir;
    benchmarkArgs.model_file = FLAGS_model_file;
    benchmarkArgs.param_file = FLAGS_param_file;
    benchmarkArgs.optimized_model_dir = FLAGS_optimized_model_dir;

    if (FLAGS_inputs_shape.empty()) {
        // show_usage();
        throw std::logic_error("Inputs shape is required but not set. Please set \
            --inputs_shape option.");
    }
    std::vector<std::string> flags_inputs_shape = Split<std::string>(FLAGS_inputs_shape, ";");
    for (int i = 0; i < flags_inputs_shape.size(); i++) {
        std::vector<std::string> input = Split<std::string>(flags_inputs_shape[i], ":");
        std::string input_name = input[0];
        std::vector<float> input_shape_value = Split<float>(input[1], ",");
        benchmarkArgs.inputs_shape[input_name] = input_shape_value;
    }

    if (FLAGS_batch_sizes.empty()) {
        // show_usage();
        throw std::logic_error("Batch sizes is required but not set. Please set \
            --batch_sizes option.");
    }
    benchmarkArgs.batch_sizes = Split<int>(FLAGS_batch_sizes, ",");

    if (FLAGS_dataset_dir.empty()) {
        // show_usage();
        throw std::logic_error("Dataset dir is required but not set. Please set \
            --dataset_dir option.");
    }
    benchmarkArgs.dataset_dir = FLAGS_dataset_dir;

    if (FLAGS_metrics.empty()) {
        // show_usage();
        throw std::logic_error("Metrics is required but not set. Please set \
            --metrics option.");
    }
    benchmarkArgs.metrics = Split<std::string>(FLAGS_metrics, ","); ;

    if (FLAGS_nnadapter_device_names.empty()) {
        // show_usage();
        throw std::logic_error("NNAdapter device names is required but not set. Please set \
            --nnadapter_device_names option.");
    }
    benchmarkArgs.nnadapter_device_names = Split<std::string>(FLAGS_nnadapter_device_names, ",");

    if (!FLAGS_nnadapter_context_properties.empty()) {
        benchmarkArgs.nnadapter_context_properties = FLAGS_nnadapter_context_properties;
    }
    

    if (FLAGS_nnadapter_subgraph_partition_config_path.empty()) {
      benchmarkArgs.nnadapter_subgraph_partition_config_path = FLAGS_nnadapter_subgraph_partition_config_path;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
  BenchmarkArgs benchmarkArgs;
  CheckAndParserArgs(argc, argv, benchmarkArgs);
  BenchmarkDemo benchmarkDemo(benchmarkArgs);
  benchmarkDemo.InitMetrics();
  benchmarkDemo.ReadDatasetLabels();
  benchmarkDemo.Test();
  benchmarkDemo.ShowMetrics();
  return 0;
}