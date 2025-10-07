# scripts/benchmark_gpu.py
import subprocess
import time
from pathlib import Path

MODELS = {
    "FP16": "models/llama-2-7b-fp16.gguf",
    "Q4_K_M": "models/llama-2-7b-q4_K_M.gguf"
}

LLAMA_BIN = "llama.cpp/build/bin/llama-gguf"  # or llama-cli if you prefer
#LLAMA_BIN = "llama.cpp/build/bin/llama" 
#LLAMA_BIN = "llama.cpp/main"  # llama.cpp GPU binary
LOG_PATH = Path("logs/gpu_benchmark.log")
PROMPTS = [
    "Explain artificial intelligence in simple words.",
    "Summarize Python programming language.",
    "Write a short story about a robot."
]

with open(LOG_PATH, "w") as log_file:
    log_file.write("GPU Benchmark Log\n")
    log_file.write("="*50 + "\n")
    for name, model_path in MODELS.items():
        log_file.write(f"Model: {name}\n")
        for prompt in PROMPTS:
            print(f"Running {name} benchmark for prompt: {prompt}")
            start = time.time()
            result = subprocess.run([
                LLAMA_BIN,
                "-m", model_path,
                "-p", prompt,
                "--gpu"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = time.time() - start
            output = result.stdout.decode("utf-8", errors="ignore")
            log_file.write(f"Prompt: {prompt}\nTime: {elapsed:.2f}s\nOutput:\n{output}\n{'-'*50}\n")
            print(f"Done in {elapsed:.2f}s\n")
