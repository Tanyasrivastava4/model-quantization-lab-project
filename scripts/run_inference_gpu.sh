#!/bin/bash
# scripts/run_inference_gpu.sh

MODEL_PATH="models/llama-2-7b-q4_K_M.gguf"
LLAMA_BIN="llama.cpp/build/bin/llama-cli"

if [ -z "$1" ]; then
    echo "Usage: ./run_inference_gpu.sh 'Your prompt here'"
    exit 1
fi

PROMPT="$1"

echo "Running inference on GPU..."
$LLAMA_BIN -m $MODEL_PATH -p "$PROMPT" 







#!/bin/bash
# scripts/run_inference_gpu.sh
#MODEL_PATH="../models/llama-2-7b-q4_K_M.gguf"
#LLAMA_BIN="../llama.cpp/main"

#if [ -z "$1" ]; then
 #   echo "Usage: ./run_inference_gpu.sh 'Your prompt here'"
  #  exit 1
#fi

#PROMPT="$1"

#echo "Running inference on GPU..."
#$LLAMA_BIN -m $MODEL_PATH -p "$PROMPT" --gpu
