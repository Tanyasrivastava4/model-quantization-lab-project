#!/bin/bash
# Run all prompts on FP16 and Q4_K_M models

PROMPTS_FILE="prompts.txt"
FP16_OUTPUT="outputs_fp16.txt"
Q4_OUTPUT="outputs_q4.txt"

# Clear previous outputs
> $FP16_OUTPUT
> $Q4_OUTPUT

MODELS=("models/llama-2-7b-fp16.gguf" "models/llama-2-7b-q4_K_M.gguf")
OUTPUTS=($FP16_OUTPUT $Q4_OUTPUT)

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    OUTPUT_FILE=${OUTPUTS[$i]}

    while IFS= read -r prompt; do
        echo "Prompt: $prompt" | tee -a $OUTPUT_FILE
        START=$(date +%s)
        ./scripts/run_inference_gpu.sh "$prompt" >> $OUTPUT_FILE 2>&1
        END=$(date +%s)
        ELAPSED=$((END-START))
        echo "Elapsed_time_sec: $ELAPSED" >> $OUTPUT_FILE
        echo "----------------------------------------" >> $OUTPUT_FILE
    done < $PROMPTS_FILE
done
