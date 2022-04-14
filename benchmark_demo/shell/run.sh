#!/bin/bash
MODEL_NAME=ResNet50
if [ -n "$1" ]; then
  MODEL_NAME=$1
fi

TARGET_OS=linux
if [ -n "$2" ]; then
  TARGET_OS=$2
fi

TARGET_ABI=amd64
if [ -n "$3" ]; then
  TARGET_ABI=$3
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}

NNADAPTER_DEVICE_NAME="nvidia_tensorrt"
NVIDIA_TENSORRT_DEVICE_TYPE=GPU
NVIDIA_TENSORRT_DEVICE_ID=0
NVIDIA_TENSORRT_PRECISION=float16

NNADAPTER_CONTEXT_PROPERTIES="NVIDIA_TENSORRT_DEVICE_TYPE=$NVIDIA_TENSORRT_DEVICE_TYPE;NVIDIA_TENSORRT_DEVICE_ID=$NVIDIA_TENSORRT_DEVICE_ID;NVIDIA_TENSORRT_PRECISION=$NVIDIA_TENSORRT_PRECISION"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/cpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/$NNADAPTER_DEVICE_NAME

if [ -e output/$MODEL_NAME.txt ]; then 
  rm -rf output/$MODEL_NAME.txt
fi

./${BUILD_DIR}/benchmark_demo \
    --model_name=$MODEL_NAME \
    --model_file=../assets/models/$MODEL_NAME/inference.pdmodel \
    --param_file=../assets/models/$MODEL_NAME/inference.pdiparams \
    --optimized_model_dir=../assets/models/$MODEL_NAME \
    --dataset_dir=../assets/datasets/imagenet \
    --nnadapter_device_names=${NNADAPTER_DEVICE_NAME} \
    --nnadapter_context_properties=${NNADAPTER_CONTEXT_PROPERTIES} \
    --inputs_shape="x:-1,3,224,224" \
    --batch_sizes="1,2,4,8,16,32" \
    --metrics="topk,latency,throughput" \


NVIDIA_TENSORRT_DEVICE_TYPE=DLA
NVIDIA_TENSORRT_DEVICE_ID=0
NVIDIA_TENSORRT_PRECISION=float16
NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/datasets/imagenet/data
NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/$MODEL_NAME/calibration_table
NNADAPTER_CONTEXT_PROPERTIES="NVIDIA_TENSORRT_DEVICE_TYPE=$NVIDIA_TENSORRT_DEVICE_TYPE;NVIDIA_TENSORRT_DEVICE_ID=$NVIDIA_TENSORRT_DEVICE_ID;NVIDIA_TENSORRT_PRECISION=$NVIDIA_TENSORRT_PRECISION"

./${BUILD_DIR}/benchmark_demo \
    --model_name=$MODEL_NAME \
    --model_file=../assets/models/$MODEL_NAME/inference.pdmodel \
    --param_file=../assets/models/$MODEL_NAME/inference.pdiparams \
    --optimized_model_dir=../assets/models/$MODEL_NAME \
    --dataset_dir=../assets/datasets/imagenet \
    --nnadapter_device_names=${NNADAPTER_DEVICE_NAME} \
    --nnadapter_context_properties=${NNADAPTER_CONTEXT_PROPERTIES} \
    --inputs_shape="x:-1,3,224,224" \
    --batch_sizes="1,2,4,8,16,32" \
    --metrics="topk,latency,throughput" \

NVIDIA_TENSORRT_DEVICE_TYPE=GPU
NVIDIA_TENSORRT_DEVICE_ID=0
NVIDIA_TENSORRT_PRECISION=int8
NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/datasets/imagenet/data
NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/$MODEL_NAME/calibration_table
NNADAPTER_CONTEXT_PROPERTIES="NVIDIA_TENSORRT_DEVICE_TYPE=$NVIDIA_TENSORRT_DEVICE_TYPE;NVIDIA_TENSORRT_DEVICE_ID=$NVIDIA_TENSORRT_DEVICE_ID;NVIDIA_TENSORRT_PRECISION=$NVIDIA_TENSORRT_PRECISION;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=$NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=$NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH"

./${BUILD_DIR}/benchmark_demo \
    --model_name=$MODEL_NAME \
    --model_file=../assets/models/$MODEL_NAME/inference.pdmodel \
    --param_file=../assets/models/$MODEL_NAME/inference.pdiparams \
    --optimized_model_dir=../assets/models/$MODEL_NAME \
    --dataset_dir=../assets/datasets/imagenet \
    --nnadapter_device_names=${NNADAPTER_DEVICE_NAME} \
    --nnadapter_context_properties=${NNADAPTER_CONTEXT_PROPERTIES} \
    --inputs_shape="x:-1,3,224,224" \
    --batch_sizes="1,2,4,8,16,32" \
    --metrics="topk,latency,throughput" \

NVIDIA_TENSORRT_DEVICE_TYPE=DLA
NVIDIA_TENSORRT_DEVICE_ID=0
NVIDIA_TENSORRT_PRECISION=int8
NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=../assets/datasets/imagenet/data
NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=../assets/models/$MODEL_NAME/calibration_table
NNADAPTER_CONTEXT_PROPERTIES="NVIDIA_TENSORRT_DEVICE_TYPE=$NVIDIA_TENSORRT_DEVICE_TYPE;NVIDIA_TENSORRT_DEVICE_ID=$NVIDIA_TENSORRT_DEVICE_ID;NVIDIA_TENSORRT_PRECISION=$NVIDIA_TENSORRT_PRECISION;NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH=$NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH;NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH=$NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH"

./${BUILD_DIR}/benchmark_demo \
    --model_name=$MODEL_NAME \
    --model_file=../assets/models/$MODEL_NAME/inference.pdmodel \
    --param_file=../assets/models/$MODEL_NAME/inference.pdiparams \
    --optimized_model_dir=../assets/models/$MODEL_NAME \
    --dataset_dir=../assets/datasets/imagenet \
    --nnadapter_device_names=${NNADAPTER_DEVICE_NAME} \
    --nnadapter_context_properties=${NNADAPTER_CONTEXT_PROPERTIES} \
    --inputs_shape="x:-1,3,224,224" \
    --batch_sizes="1,2,4,8,16,32" \
    --metrics="topk,latency,throughput" \