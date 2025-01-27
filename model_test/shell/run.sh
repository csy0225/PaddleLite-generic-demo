#!/bin/bash
#MODEL_NAME=conv_add_relu_dwconv_add_relu_224_int8_per_layer
#MODEL_NAME=conv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x4_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_mul_add_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_conv_add_relu_dwconv_add_relu_224_int8_per_channel
#MODEL_NAME=conv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_mul_add_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_res2c_224_fp32
#INPUT_SHAPE="1,3,224,224"
#INPUT_TYPE="float32"
#OUTPUT_TYPE="float32"
#MODEL_TYPE=0

#MODEL_NAME=conv_add_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_mul_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_mul_144_192_int8_per_layer
#INPUT_SHAPE="1,3,192,144"
#INPUT_TYPE="float32"
#OUTPUT_TYPE="float32"
#MODEL_TYPE=0

#MODEL_NAME=eltwise_mul_broadcast_per_layer
#INPUT_SHAPE="1,3,384,384"
#INPUT_TYPE="float32"
#OUTPUT_TYPE="float32"
#MODEL_TYPE=0

#MODEL_NAME=dwconv_ic_128_groups_128_oc_256_per_layer
#INPUT_SHAPE="1,3,320,320"
#INPUT_TYPE="float32"
#OUTPUT_TYPE="float32"
#MODEL_TYPE=0

if [ -n "$1" ]; then
    MODEL_NAME=$1
fi

if [ -n "$2" ]; then
    MODEL_TYPE=$2
fi

if [ -n "$3" ]; then
    INPUT_SHAPE=$3
fi

if [ -n "$4" ]; then
    INPUT_TYPE=$4
fi

if [ -n "$5" ]; then
    OUTPUT_TYPE=$5
fi

#SUBGRAPH_PARTITION_CONFIG_FILE=subgraph_partition_config_file.txt
WORK_SPACE="~/test"

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=arm64-v8a
# MT8168/8175, Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=armeabi-v7a
# RK1808EVB, TB-RK1808S0, Kunpeng-920+Ascend310: TARGET_OS=linux and TARGET_ABI=arm64
# RK1806EVB, RV1109/1126 EVB: TARGET_OS=linux and TARGET_ABI=armhf 
# Intel-x86+Ascend310: TARGET_OS=linux and TARGET_ABI=amd64
# Intel-x86+CambriconMLU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
if [ -n "$6" ]; then
    TARGET_OS=$6
fi

TARGET_ABI=arm64
if [ -n "$7" ]; then
    TARGET_ABI=$7
fi

# RK1808EVB, TB-RK1808S0, RK1806EVB, RV1109/1126 EVB: NNADAPTER_DEVICE_NAMES=rockchip_npu
# MT8168/8175: NNADAPTER_DEVICE_NAMES=mediatek_apu
# Kirin810/820/985/990/9000/9000E: NNADAPTER_DEVICE_NAMES=huawei_kirin_npu
# Ascend310: NNADAPTER_DEVICE_NAMES=huawei_ascend_npu
# CambriconMLU: NNADAPTER_DEVICE_NAMES=cambricon_mlu
# CPU only: NNADAPTER_DEVICE_NAMES=cpu
NNADAPTER_DEVICE_NAMES="cpu"
if [ -n "$8" ]; then
    NNADAPTER_DEVICE_NAMES="$8"
fi

NNADAPTER_CONTEXT_PROPERTIES="null"
if [ -n "$9" ]; then
    NNADAPTER_CONTEXT_PROPERTIES="$9"
fi

NNADAPTER_MODEL_CACHE_DIR="null"
#NNADAPTER_MODEL_CACHE_DIR="."
if [ -n "${10}" ]; then
    NNADAPTER_MODEL_CACHE_DIR="${10}"
fi

NNADAPTER_MODEL_CACHE_TOKEN="null"
if [ -n "${11}" ]; then
    NNADAPTER_MODEL_CACHE_TOKEN="${11}"
fi

NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="null"
if [ -n "$SUBGRAPH_PARTITION_CONFIG_FILE" ]; then
    NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="../assets/models/$MODEL_NAME/$SUBGRAPH_PARTITION_CONFIG_FILE"
fi

export GLOG_v=5
export SUBGRAPH_ONLINE_MODE=true
if [ "$NNADAPTER_DEVICE_NAMES" == "rockchip_npu" ]; then
  export RKNPU_LOGLEVEL=5
  export RKNN_LOG_LEVEL=5
  ulimit -c unlimited
  echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  echo $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
fi

if [ "$NNADAPTER_DEVICE_NAMES" == "amlogic_npu" ]; then
  echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
fi

if [ "$NNADAPTER_DEVICE_NAMES" == "imagination_nna" ]; then
  echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/$NNADAPTER_DEVICE_NAMES::../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/cpu
if [ "$NNADAPTER_DEVICE_NAMES" == "huawei_ascend_npu" ]; then
    HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
    if [ "$TARGET_OS" == "linux" ]; then
      if [[ "$TARGET_ABI" != "arm64" && "$TARGET_ABI" != "amd64" ]]; then
        echo "Unknown OS $TARGET_OS, only supports 'arm64' or 'amd64' for Huawei Ascend NPU."
        exit -1
      fi
    else
      echo "Unknown OS $TARGET_OS, only supports 'linux' for Huawei Ascend NPU."
      exit -1
    fi
    NNADAPTER_CONTEXT_PROPERTIES="HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/stub:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/opp/op_proto/built-in
    export PYTHONPATH=$PYTHONPATH:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/pyACL/python/site-packages/acl
    export PATH=$PATH:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/ccec_compiler/bin:${HUAWEI_ASCEND_TOOLKIT_HOME}/acllib/bin:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/bin
    export ASCEND_AICPU_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME
    export ASCEND_OPP_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME/opp
    export TOOLCHAIN_HOME=$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=3
fi

if [ "$NNADAPTER_DEVICE_NAMES" == "kunlunxin_xtcl" ]; then
    export XTCL_AUTO_ALLOC_L3=1
    export XTCL_CONV_USE_FP16=1
    export XTCL_QUANTIZE_WEIGHT=1
    export XTCL_L3_SIZE=16777216
fi

if [ "$NNADAPTER_DEVICE_NAMES" == "cambricon_mlu" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/neuware/lib64"
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}

set -e
./$BUILD_DIR/model_test ../assets/models/$MODEL_NAME $MODEL_TYPE $INPUT_SHAPE $INPUT_TYPE $OUTPUT_TYPE $NNADAPTER_DEVICE_NAMES $NNADAPTER_CONTEXT_PROPERTIES $NNADAPTER_MODEL_CACHE_DIR $NNADAPTER_MODEL_CACHE_TOKEN $NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH
