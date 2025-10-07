# scripts/convert_hf_to_gguf.py
#import os
#import subprocess
#from pathlib import Path

# Paths
#HF_MODEL = "facebook/llama-2-7b-hf"  # Hugging Face model repo
#OUTPUT_DIR = Path("../models")
#TEMP_DIR = Path("../temp/hf_weights")
#OUTPUT_DIR.mkdir(exist_ok=True)
#TEMP_DIR.mkdir(exist_ok=True)

#TEMP_DIR = Path("temp/hf_weights")  # remove the ../
#OUTPUT_DIR = Path("models")         # same for output if you want
#OUTPUT_DIR.mkdir(exist_ok=True)
#TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Download Hugging Face weights (if not already downloaded)
#print("Downloading Hugging Face model weights...")
#subprocess.run([
 #   "python", "-m", "transformers-cli", "download",
  #  HF_MODEL, "--cache-dir", str(TEMP_DIR)
#])

# Step 2: Convert HF weights → FP16 GGUF using llama.cpp
#print("Converting HF weights to FP16 GGUF format...")
#subprocess.run([
 #   "python", "../llama.cpp/tools/convert-hf-to-gguf.py",
 #   "--input", str(TEMP_DIR / HF_MODEL.split("/")[-1]),
 #   "--output", str(OUTPUT_DIR / "llama-2-7b-fp16.gguf"),
 #   "--dtype", "fp16"
#])

#print("Conversion completed. FP16 GGUF saved in 'models/'")




#HF_MODEL = "facebook/llama-2-7b-hf"  # Hugging Face model repo
#HF_MODEL = "meta-llama/Llama-2-7b-hf"      # or Llama-2-7b-chat-hf
#OUTPUT_DIR = Path("models")          # Where FP16 GGUF will be saved
#TEMP_DIR = Path("temp/hf_weights")   # Temp folder for HF weights
#LLAMA_CPP_CONVERT = Path("llama.cpp/tools/convert-hf-to-gguf.py")  # Adjust if your path is different

# scripts/convert_hf_to_gguf.py
import os
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Config: Hugging Face model repo
HF_MODEL = "meta-llama/Llama-2-7b-hf"

# Paths
OUTPUT_DIR = Path("models")
TEMP_DIR = Path("temp/hf_weights")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# -------------------------------
# Step 1: Ensure Hugging Face login
print("Make sure you have run 'huggingface-cli login' with your token.")

# -------------------------------
# Step 2: Download Hugging Face weights using Transformers
print("Downloading Hugging Face model weights (this may take a while)...")

try:
    # Auto-download using Transformers
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, cache_dir=TEMP_DIR, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, cache_dir=TEMP_DIR, use_auth_token=True)
    print("Hugging Face model weights downloaded successfully.")
except Exception as e:
    print("Error downloading weights:", e)
    print("Make sure you accepted the LLaMA-2 license on Hugging Face.")
    exit(1)

# -------------------------------
# Step 3: Convert HF weights → FP16 GGUF using llama.cpp
print("Converting HF weights to FP16 GGUF format...")

# Path to llama.cpp conversion script
CONVERT_SCRIPT = "../llama.cpp/tools/convert-hf-to-gguf.py"

# Local folder where HF weights are cached
HF_LOCAL_PATH = TEMP_DIR / HF_MODEL.split("/")[-1]

OUTPUT_FP16 = OUTPUT_DIR / "llama-2-7b-fp16.gguf"

subprocess.run([
    "python", CONVERT_SCRIPT,
    "--input", str(HF_LOCAL_PATH),
    "--output", str(OUTPUT_FP16),
    "--dtype", "fp16"
])

print(f"Conversion completed. FP16 GGUF saved in '{OUTPUT_DIR}'")

