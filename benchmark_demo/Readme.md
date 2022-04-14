# Benchmark Demo
## 测试内容
1. 指标选择，作为一个可选项，可扩展，
2. batch数目选择，支持列表输入，已逗号为分隔符
3. 支持设备选择，自动产生报表

命令行格式：
    ./benchmark
    --model_name=""
    --model_dir=""
    --nnadapter_device_names="nvidia_tensorrt"
    --nnadapter_context_properties=""
    --inputs_shape="im_shape:-1,2;image:-1,3,224,224;scale_factor:-1,2"
    --batch_sizes="1,2,4,8,16,32"
    --metrics="topk,latency"



# 会产生一个报告，列表里面记录了这次benchmark配置的信息。可以用json实现或者直接写文件。