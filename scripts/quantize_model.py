# scripts/quantize_model.py
import subprocess
from pathlib import Path

FP16_MODEL = Path("../models/llama-2-7b-fp16.gguf")
OUTPUT_MODEL = Path("../models/llama-2-7b-q4_K_M.gguf")

print("Quantizing FP16 model to 4-bit Q4_K_M...")
subprocess.run([
    "../llama.cpp/quantize",
    str(FP16_MODEL),
    str(OUTPUT_MODEL),
    "q4_K_M"
])

print("Quantization completed. Q4_K_M model saved in 'models/'")
